# -*- coding: utf-8 -*-
"""Fine-tune a multilingual transformer.

   python scripts/run_transformer.py [model_name] [emoji_mode] [epochs] [max_len] [batch]
          [--lora R] [--lr LR] [--smooth S] [--no-freeze] [--tag NAME] [--accum N]

Writes outputs/pred_test_<tag>.csv, pred_val_<tag>.csv, errors_<tag>.csv, figures/cm_<tag>.png,
appends to outputs/results_transformer_test.csv and saves the model under outputs/model_<tag>/.
"""
import argparse
import faulthandler
import json
import os
import sys
import time

faulthandler.enable()

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluate import error_table, evaluate, plot_confusion, results_table
from src.neural import best_threshold
from src.transformer import fine_tune, predict_texts

ap = argparse.ArgumentParser()
ap.add_argument("model", nargs="?", default="xlm-roberta-base")
ap.add_argument("emoji_mode", nargs="?", default="keep")
ap.add_argument("epochs", nargs="?", type=int, default=3)
ap.add_argument("max_len", nargs="?", type=int, default=96)
ap.add_argument("bs", nargs="?", type=int, default=16)
ap.add_argument("--lora", type=int, default=0)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--smooth", type=float, default=0.0)
ap.add_argument("--accum", type=int, default=1)
ap.add_argument("--no-freeze", action="store_true")
ap.add_argument("--fp32", action="store_true", help="disable mixed precision (DeBERTa-v3 produces NaN under autocast)")
ap.add_argument("--tag", default=None)
a = ap.parse_args()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs"); FIG = os.path.join(OUT, "figures"); SPL = os.path.join(OUT, "splits")
train = pd.read_csv(os.path.join(SPL, "train.csv")); val = pd.read_csv(os.path.join(SPL, "val.csv")); test = pd.read_csv(os.path.join(SPL, "test.csv"))
y_tr, y_va, y_te = (d["is_sarcastic"].values for d in (train, val, test))
tag = a.tag or (a.model.split("/")[-1] + "_" + a.emoji_mode + (f"_lora{a.lora}" if a.lora else ""))
print("cuda:", torch.cuda.is_available(), "| model:", a.model, "| emoji:", a.emoji_mode, "| tag:", tag, "| lora:", a.lora, "| lr:", a.lr, flush=True)
t0 = time.time()
model, tok, hist, p_va = fine_tune(a.model, train["urdu_text"], y_tr, val["urdu_text"], y_va, emoji_mode=a.emoji_mode,
                                   epochs=a.epochs, max_len=a.max_len, bs=a.bs, lr=a.lr, grad_accum=a.accum,
                                   freeze_embeddings=not a.no_freeze, lora_r=a.lora, label_smoothing=a.smooth,
                                   fp16=False if a.fp32 else None)
print("training finished; tuning threshold", flush=True)
thr, f_va = best_threshold(y_va, p_va)
p_te = predict_texts(model, tok, test["urdu_text"], emoji_mode=a.emoji_mode, max_len=a.max_len)
pred = (p_te >= thr).astype(int)
print("test predictions done", flush=True)
name = f"{a.model} (emoji {a.emoji_mode}{', LoRA r=' + str(a.lora) if a.lora else ''})"
r = evaluate(y_te, pred, name); r.update(threshold=thr, val_macro_f1=f_va, epochs=len(hist), emoji_mode=a.emoji_mode, model_name=a.model, tag=tag)
pd.DataFrame({"urdu_text": test["urdu_text"], "is_sarcastic": y_te, "pred": pred, "p_sarcastic": p_te}).to_csv(
    os.path.join(OUT, f"pred_test_{tag}.csv"), index=False, encoding="utf-8")
pd.DataFrame({"urdu_text": val["urdu_text"], "is_sarcastic": y_va, "p_sarcastic": p_va}).to_csv(
    os.path.join(OUT, f"pred_val_{tag}.csv"), index=False, encoding="utf-8")
plot_confusion(r, os.path.join(FIG, f"cm_{tag}.png"))
error_table(test, pred, p_te, k=25).to_csv(os.path.join(OUT, f"errors_{tag}.csv"), index=False, encoding="utf-8")
path = os.path.join(OUT, "results_transformer_test.csv")
old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
res_df = pd.concat([old, results_table([r])]).drop_duplicates("model", keep="last")
res_df.to_csv(path, index=False)
json.dump({"test": r, "history": hist, "args": vars(a), "elapsed_s": round(time.time() - t0)},
          open(os.path.join(OUT, f"transformer_{tag}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
save_dir = os.path.join(OUT, f"model_{tag}")
try:
    model.save_pretrained(save_dir, safe_serialization=False); tok.save_pretrained(save_dir)
    print("model saved to", save_dir, flush=True)
except Exception as ex:
    print("model save failed:", ex, flush=True)
print(f"==> {name}: test macro-F1 {r['macro_f1']:.4f} acc {r['accuracy']:.4f} (thr {thr:.2f}, val F1 {f_va:.4f}) in {time.time() - t0:.0f}s", flush=True)
