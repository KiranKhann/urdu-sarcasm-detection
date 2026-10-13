# -*- coding: utf-8 -*-
"""Fine-tune a multilingual transformer.  Usage:
   python scripts/run_transformer.py [model_name] [emoji_mode] [epochs] [max_len] [batch]
"""
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs"); FIG = os.path.join(OUT, "figures"); SPL = os.path.join(OUT, "splits")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "xlm-roberta-base"
EMOJI_MODE = sys.argv[2] if len(sys.argv) > 2 else "keep"
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
MAX_LEN = int(sys.argv[4]) if len(sys.argv) > 4 else 96
BS = int(sys.argv[5]) if len(sys.argv) > 5 else 16

train = pd.read_csv(os.path.join(SPL, "train.csv")); val = pd.read_csv(os.path.join(SPL, "val.csv")); test = pd.read_csv(os.path.join(SPL, "test.csv"))
y_tr, y_va, y_te = (d["is_sarcastic"].values for d in (train, val, test))
print("cuda:", torch.cuda.is_available(), "| model:", MODEL, "| emoji:", EMOJI_MODE)
t0 = time.time()
model, tok, hist, p_va = fine_tune(MODEL, train["urdu_text"], y_tr, val["urdu_text"], y_va, emoji_mode=EMOJI_MODE,
                                   epochs=EPOCHS, max_len=MAX_LEN, bs=BS)
print("training finished; tuning threshold", flush=True)
thr, f_va = best_threshold(y_va, p_va)
p_te = predict_texts(model, tok, test["urdu_text"], emoji_mode=EMOJI_MODE, max_len=MAX_LEN)
pred = (p_te >= thr).astype(int)
print("test predictions done", flush=True)
name = f"{MODEL} (emoji {EMOJI_MODE})"
r = evaluate(y_te, pred, name); r.update(threshold=thr, val_macro_f1=f_va, epochs=len(hist), emoji_mode=EMOJI_MODE, model_name=MODEL)
tag = MODEL.split("/")[-1] + "_" + EMOJI_MODE
pd.DataFrame({"urdu_text": test["urdu_text"], "is_sarcastic": y_te, "pred": pred, "p_sarcastic": p_te}).to_csv(
    os.path.join(OUT, f"pred_test_{tag}.csv"), index=False, encoding="utf-8")
plot_confusion(r, os.path.join(FIG, f"cm_{tag}.png"))
error_table(test, pred, p_te, k=25).to_csv(os.path.join(OUT, f"errors_{tag}.csv"), index=False, encoding="utf-8")
path = os.path.join(OUT, "results_transformer_test.csv")
old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
res_df = pd.concat([old, results_table([r])]).drop_duplicates("model", keep="last")
res_df.to_csv(path, index=False)
json.dump({"test": r, "history": hist, "elapsed_s": round(time.time() - t0)},
          open(os.path.join(OUT, f"transformer_{tag}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
save_dir = os.path.join(OUT, f"model_{tag}")
try:
    model.save_pretrained(save_dir, safe_serialization=False); tok.save_pretrained(save_dir)
    print("model saved to", save_dir, flush=True)
except Exception as ex:
    print("model save failed:", ex, flush=True)
print(f"==> {name}: test macro-F1 {r['macro_f1']:.4f} acc {r['accuracy']:.4f} (thr {thr:.2f}) in {time.time() - t0:.0f}s")
