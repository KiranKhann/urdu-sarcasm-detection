# -*- coding: utf-8 -*-
"""
Ensemble of the strongest models.  Probabilities are averaged (or weighted by a non-negative
least-squares fit on the VALIDATION split) and the decision threshold is tuned on validation only.
Member validation probabilities are produced here if they were not saved by the training scripts.

   python scripts/run_ensemble.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluate import evaluate, plot_confusion, results_table
from src.neural import best_threshold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs"); FIG = os.path.join(OUT, "figures"); SPL = os.path.join(OUT, "splits")
train = pd.read_csv(os.path.join(SPL, "train.csv")); val = pd.read_csv(os.path.join(SPL, "val.csv")); test = pd.read_csv(os.path.join(SPL, "test.csv"))
y_tr, y_va, y_te = (d["is_sarcastic"].values for d in (train, val, test))

members = {}   # tag -> (val_probs, test_probs)


def add(tag, pv, pt):
    pv, pt = np.asarray(pv, float), np.asarray(pt, float)
    if np.isnan(pv).any() or np.isnan(pt).any() or best_threshold(y_va, pv)[1] < 0.6:
        print(f"  member {tag:40s} SKIPPED (degenerate: NaN or validation macro-F1 < 0.6)")
        return
    members[tag] = (pv, pt)
    r = evaluate(y_te, (pt >= best_threshold(y_va, pv)[0]).astype(int))
    print(f"  member {tag:40s} test macro-F1 {r['macro_f1']:.4f}  acc {r['accuracy']:.4f}")


# ---- transformers: use saved val probs, or compute them from the saved model ----------------------
for ptest in sorted(glob.glob(os.path.join(OUT, "pred_test_*.csv"))):
    tag = os.path.basename(ptest)[len("pred_test_"):-4]
    if not (tag.startswith(("xlm-roberta", "muril", "mdeberta"))):
        continue
    pval = os.path.join(OUT, f"pred_val_{tag}.csv")
    pt = pd.read_csv(ptest)["p_sarcastic"].values
    if os.path.exists(pval):
        pv = pd.read_csv(pval)["p_sarcastic"].values
    else:
        mdir = os.path.join(OUT, f"model_{tag}")
        if not os.path.isdir(mdir):
            continue
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from src.transformer import predict_texts
        emoji_mode = tag.split("_")[1]
        tok = AutoTokenizer.from_pretrained(mdir)
        model = AutoModelForSequenceClassification.from_pretrained(mdir).to("cuda" if torch.cuda.is_available() else "cpu").eval()
        pv = predict_texts(model, tok, val["urdu_text"], emoji_mode=emoji_mode, max_len=96)
        pd.DataFrame({"urdu_text": val["urdu_text"], "is_sarcastic": y_va, "p_sarcastic": pv}).to_csv(pval, index=False, encoding="utf-8")
        del model; torch.cuda.empty_cache()
    add(tag, pv, pt)

# ---- classical LR (refit, 30 s) and TextCNN hybrid (retrain, ~1 min) --------------------------------
from src.baselines import make_pipeline, predict_proba
pipe = make_pipeline("word+char+hand", clf="lr", C=2.0, emoji_mode="both").fit(train["urdu_text"].tolist(), y_tr)
add("tfidf_lr", predict_proba(pipe, val["urdu_text"].tolist()), predict_proba(pipe, test["urdu_text"].tolist()))

import torch
from src.features import HandcraftedFeatures
from src.neural import TextCNN, Vocab, get_device, predict, set_seed, train_model
set_seed(42)
vocab = Vocab(min_freq=2, emoji_mode="keep").fit(train["urdu_text"])
enc = lambda d: np.array([vocab.encode(t, 64) for t in d["urdu_text"]], dtype=np.int64)
hf = HandcraftedFeatures().fit(train["urdu_text"], y_tr); D = hf.transform(train["urdu_text"]); mu, sd = D.mean(0), D.std(0) + 1e-6
Dn = lambda d: ((hf.transform(d["urdu_text"]) - mu) / sd).astype(np.float32)
cnn = TextCNN(len(vocab), n_dense=D.shape[1])
cnn, _ = train_model(cnn, enc(train), y_tr, enc(val), y_va, Dn(train), Dn(val), epochs=20, patience=4, verbose=False, name="cnn")
add("textcnn_hybrid", predict(cnn, enc(val), Dn(val)), predict(cnn, enc(test), Dn(test)))
# persist the members that the deployable manifest may reference
import joblib
joblib.dump(pipe, os.path.join(OUT, "model_tfidf_lr.joblib"))
torch.save({"state": {k: v.cpu() for k, v in cnn.state_dict().items()}, "itos": vocab.itos, "emoji_mode": "keep",
            "max_len": 64, "hf": hf, "mu": mu, "sd": sd, "n_dense": int(D.shape[1])}, os.path.join(OUT, "model_textcnn_hybrid.pt"))

# ---- ensembles ----------------------------------------------------------------------------------------
tags = list(members)
PV = np.column_stack([members[t][0] for t in tags]); PT = np.column_stack([members[t][1] for t in tags])
results = []


def report(name, pv, pt, weights):
    """weights: dict tag -> weight (normalised to sum 1)."""
    thr, f_va = best_threshold(y_va, pv)
    r = evaluate(y_te, (pt >= thr).astype(int), name)
    r.update(threshold=thr, val_macro_f1=f_va, weights={t: round(float(x), 4) for t, x in weights.items() if x > 1e-4})
    results.append(r)
    print(f"{name:60s} val F1 {f_va:.4f}  test macro-F1 {r['macro_f1']:.4f}  acc {r['accuracy']:.4f}")
    return r


def combine(wdict):
    w = np.array([wdict.get(t, 0.0) for t in tags], float); w = w / w.sum()
    return PV @ w, PT @ w, dict(zip(tags, w))


print("\n=== single models (threshold tuned on validation) ===")
for t in tags:
    report(f"Single: {t}", *combine({t: 1.0}))
print("\n=== ensembles ===")
report("Ensemble: mean of all members", *combine({t: 1.0 for t in tags}))
tr_tags = [t for t in tags if t.startswith(("xlm-roberta", "muril", "mdeberta"))]
if len(tr_tags) > 1:
    report("Ensemble: mean of transformers", *combine({t: 1.0 for t in tr_tags}))
# weights fitted on validation (non-negative least squares on the labels), then applied once to test
from scipy.optimize import nnls
w, _ = nnls(PV, y_va.astype(float))
if w.sum() <= 0:
    w = np.ones(len(tags))
report("Ensemble: validation-weighted (NNLS)", *combine(dict(zip(tags, w))))
# greedy forward selection on validation (equal weights)
sel, cur = [], None
for _ in range(len(tags)):
    cand = [(best_threshold(y_va, PV[:, sel + [j]].mean(1))[1], j) for j in range(len(tags)) if j not in sel]
    f, j = max(cand)
    if cur is not None and f <= cur + 1e-4:
        break
    sel.append(j); cur = f
report("Ensemble: greedy-selected (" + " + ".join(tags[j] for j in sel) + ")", *combine({tags[j]: 1.0 for j in sel}))
# a light "deployable" ensemble: best single transformer + TF-IDF LR + TextCNN (one encoder to ship)
best_tr = max(tr_tags, key=lambda t: best_threshold(y_va, members[t][0])[1]) if tr_tags else None
if best_tr:
    report(f"Ensemble: deployable ({best_tr} + tfidf_lr + textcnn_hybrid)",
           *combine({best_tr: 1.0, "tfidf_lr": 1.0, "textcnn_hybrid": 1.0}))

best_r = max(results, key=lambda r: r["val_macro_f1"])
print(f"\nbest by validation: {best_r['model']}  ->  test macro-F1 {best_r['macro_f1']:.4f}  acc {best_r['accuracy']:.4f}")
plot_confusion(best_r, os.path.join(FIG, "cm_ensemble.png"))
results_table(results).to_csv(os.path.join(OUT, "results_ensemble_test.csv"), index=False)
json.dump(results, open(os.path.join(OUT, "ensemble_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)


# ---- manifest of the best model (by validation) for src.predict.SarcasmExplainer.load_best() -----------
def member_spec(tag):
    if tag == "tfidf_lr":
        return dict(tag=tag, kind="tfidf", path="outputs/model_tfidf_lr.joblib")
    if tag == "textcnn_hybrid":
        return dict(tag=tag, kind="textcnn", path="outputs/model_textcnn_hybrid.pt")
    emoji_mode = tag.split("_")[1] if "_" in tag else "keep"
    return dict(tag=tag, kind="transformer", path=f"outputs/model_{tag}", emoji_mode=emoji_mode, max_len=96)


manifest = {"name": best_r["model"], "selected_by": "validation macro-F1", "val_macro_f1": best_r["val_macro_f1"],
            "test_macro_f1": best_r["macro_f1"], "test_accuracy": best_r["accuracy"], "threshold": best_r["threshold"],
            "members": [dict(member_spec(t), weight=w) for t, w in best_r["weights"].items()]}
json.dump(manifest, open(os.path.join(OUT, "best_model.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print("manifest written: outputs/best_model.json ->", [(m["tag"], m["weight"]) for m in manifest["members"]])
