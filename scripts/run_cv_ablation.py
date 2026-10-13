# -*- coding: utf-8 -*-
"""5-fold cross-validated ablation (train+val pooled) so that differences of a few
tenths of a point are not mistaken for signal.  Also tests removing the
dataset-collection artefact tokens (#طنز / #مزاح / #sarcasm hashtags)."""
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.baselines import make_pipeline
from src.evaluate import evaluate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs"); SPL = os.path.join(OUT, "splits")
train = pd.read_csv(os.path.join(SPL, "train.csv")); val = pd.read_csv(os.path.join(SPL, "val.csv"))
pool = pd.concat([train, val]).reset_index(drop=True)
X, y = pool["urdu_text"].tolist(), pool["is_sarcastic"].values

ARTEFACT = re.compile(r"#?\b(طنز|مزاح|sarcasm|sarcastic|مزاحیہ|طنزیہ)\b|#\w*(sarcas)\w*", re.I)
X_noart = [ARTEFACT.sub(" ", t) for t in X]
print("posts containing artefact tokens:", sum(bool(ARTEFACT.search(t)) for t in X),
      "| sarcastic among them:", int(y[[bool(ARTEFACT.search(t)) for t in X]].sum()))

CONFIGS = [
    ("emojis removed",                  dict(features="word+char", emoji_mode="remove"), X),
    ("emojis kept raw",                 dict(features="word+char", emoji_mode="keep"), X),
    ("emojis -> meaning token",         dict(features="word+char", emoji_mode="demojize"), X),
    ("emojis -> Urdu gloss",            dict(features="word+char", emoji_mode="urdu"), X),
    ("emojis -> meaning+cat+polarity",  dict(features="word+char", emoji_mode="both"), X),
    ("emoji symbol + Urdu meaning (augment)", dict(features="word+char", emoji_mode="augment"), X),
    ("word only",                       dict(features="word", emoji_mode="both"), X),
    ("char only",                       dict(features="char", emoji_mode="both"), X),
    ("hand-crafted only",               dict(features="hand", emoji_mode="both"), X),
    ("word+char+hand",                  dict(features="word+char+hand", emoji_mode="both"), X),
    ("raw text, no normalisation",      dict(features="word+char", normalize=False, social=False, emoji_mode="keep"), X),
    ("Urdu char normalisation only",    dict(features="word+char", normalize=True, social=False, emoji_mode="keep"), X),
    ("Urdu + social normalisation",     dict(features="word+char", normalize=True, social=True, emoji_mode="keep"), X),
    ("artefact hashtags removed (word+char+hand)", dict(features="word+char+hand", emoji_mode="both"), X_noart),
]
if len(sys.argv) > 1:   # --only <substring>: run a subset and append to the existing table
    CONFIGS = [c for c in CONFIGS if sys.argv[1] in c[0]]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows = []
t0 = time.time()
for name, cfg, data in CONFIGS:
    f1s, accs = [], []
    for tr, te in skf.split(data, y):
        kw = dict(cfg); feats = kw.pop("features")
        pipe = make_pipeline(feats, clf="lr", C=2.0, **kw)
        pipe.fit([data[i] for i in tr], y[tr])
        r = evaluate(y[te], pipe.predict([data[i] for i in te]))
        f1s.append(r["macro_f1"]); accs.append(r["accuracy"])
    rows.append({"config": name, "macro_f1_mean": np.mean(f1s), "macro_f1_std": np.std(f1s),
                 "acc_mean": np.mean(accs), "folds": f1s})
    print(f"{name:45s} macro-F1 {np.mean(f1s):.4f} ± {np.std(f1s):.4f}   ({time.time() - t0:.0f}s)", flush=True)
df = pd.DataFrame(rows)
path = os.path.join(OUT, "ablations_cv.csv")
if len(sys.argv) > 1 and os.path.exists(path):
    df = pd.concat([pd.read_csv(path), df]).drop_duplicates("config", keep="last")
df.to_csv(os.path.join(OUT, "ablations_cv.csv"), index=False)
print(df[["config", "macro_f1_mean", "macro_f1_std", "acc_mean"]].to_string())
