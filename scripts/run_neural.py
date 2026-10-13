# -*- coding: utf-8 -*-
"""Neural models (TextCNN, BiLSTM+attention, hybrid with hand-crafted features)."""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluate import error_table, evaluate, plot_confusion, results_table
from src.features import HandcraftedFeatures
from src.neural import (BiLSTMAttn, TextCNN, Vocab, best_threshold, get_device,
                        predict, set_seed, train_model)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs"); FIG = os.path.join(OUT, "figures")
SPL = os.path.join(OUT, "splits")
train = pd.read_csv(os.path.join(SPL, "train.csv")); val = pd.read_csv(os.path.join(SPL, "val.csv")); test = pd.read_csv(os.path.join(SPL, "test.csv"))
y_tr, y_va, y_te = (d["is_sarcastic"].values for d in (train, val, test))
device = get_device(); print("device:", device)
MAX_LEN = 64
t0 = time.time()

results, hists = [], {}
EMOJI_MODES = ["keep", "remove"]   # neural ablation: emoji tokens kept vs removed
for emoji_mode in EMOJI_MODES:
    set_seed(42)
    vocab = Vocab(min_freq=2, emoji_mode=emoji_mode).fit(train["urdu_text"])
    enc = lambda d: np.array([vocab.encode(t, MAX_LEN) for t in d["urdu_text"]], dtype=np.int64)
    X_tr, X_va, X_te = enc(train), enc(val), enc(test)
    print(f"\n### emoji_mode={emoji_mode}  vocab={len(vocab)}")
    for arch in ["cnn", "bilstm"]:
        for hybrid in ([False, True] if emoji_mode == "keep" else [False]):
            set_seed(42)
            if hybrid:
                hf = HandcraftedFeatures().fit(train["urdu_text"], y_tr)
                D_tr = hf.transform(train["urdu_text"]); mu, sd = D_tr.mean(0), D_tr.std(0) + 1e-6
                D_tr = ((D_tr - mu) / sd).astype(np.float32)
                D_va = ((hf.transform(val["urdu_text"]) - mu) / sd).astype(np.float32)
                D_te = ((hf.transform(test["urdu_text"]) - mu) / sd).astype(np.float32)
                nd = D_tr.shape[1]
            else:
                D_tr = D_va = D_te = None; nd = 0
            model = TextCNN(len(vocab), n_dense=nd) if arch == "cnn" else BiLSTMAttn(len(vocab), n_dense=nd)
            name = f"{'TextCNN' if arch == 'cnn' else 'BiLSTM+Attn'}" + (" + hand-crafted" if hybrid else "") + (f" (emoji {emoji_mode})")
            model, hist = train_model(model, X_tr, y_tr, X_va, y_va, D_tr, D_va, epochs=20, patience=4, name=name, device=device)
            hists[name] = hist
            p_va = predict(model, X_va, D_va, device=device)
            thr, f_va = best_threshold(y_va, p_va)
            p_te = predict(model, X_te, D_te, device=device)
            pred = (p_te >= thr).astype(int)
            r = evaluate(y_te, pred, name); r.update(threshold=thr, val_macro_f1=f_va, epochs=len(hist), emoji_mode=emoji_mode, arch=arch, hybrid=hybrid)
            results.append(r)
            print(f"==> {name}: test macro-F1 {r['macro_f1']:.4f} acc {r['accuracy']:.4f} (thr {thr:.2f}, val F1 {f_va:.4f})")
            tag = f"{arch}{'_hybrid' if hybrid else ''}_{emoji_mode}"
            pd.DataFrame({"urdu_text": test["urdu_text"], "is_sarcastic": y_te, "pred": pred, "p_sarcastic": p_te}).to_csv(
                os.path.join(OUT, f"pred_test_{tag}.csv"), index=False, encoding="utf-8")
            plot_confusion(r, os.path.join(FIG, f"cm_{tag}.png"))
            if emoji_mode == "keep" and hybrid and arch == "bilstm":
                torch.save({"state": model.state_dict(), "itos": vocab.itos, "mu": mu, "sd": sd, "thr": thr}, os.path.join(OUT, "model_bilstm_hybrid.pt"))
                error_table(test, pred, p_te, k=25).to_csv(os.path.join(OUT, "errors_bilstm.csv"), index=False, encoding="utf-8")

results_table(results).to_csv(os.path.join(OUT, "results_neural_test.csv"), index=False)
json.dump({"test": results, "history": hists, "elapsed_s": round(time.time() - t0)},
          open(os.path.join(OUT, "neural_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(results_table(results).to_string()); print(f"done in {time.time() - t0:.0f}s")
