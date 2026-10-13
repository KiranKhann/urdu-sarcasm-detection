# -*- coding: utf-8 -*-
"""EDA + classical baselines + ablations.  Writes everything under outputs/."""
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.baselines import (ABLATIONS, fit_eval, make_pipeline, predict_proba,
                           run_ablations, top_features)
from src.data import load_and_split, save_splits
from src.emoji_semantics import EmojiPrior, describe
from src.evaluate import (error_table, evaluate, phenomena_summary,
                          plot_confusion, results_table)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "urdu_sarcastic_dataset.csv")
OUT = os.path.join(ROOT, "outputs")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t_start = time.time()
df, train, val, test, rep = load_and_split(DATA)
save_splits(train, val, test, os.path.join(OUT, "splits"))
print(json.dumps(rep, indent=1, ensure_ascii=False))

# ----------------------------------------------------------------------------- EDA
eda = dict(rep)
L = df["urdu_text"].str.len()
W = df["urdu_text"].str.split().str.len()
eda["chars"] = L.describe().round(1).to_dict()
eda["words"] = W.describe().round(1).to_dict()
has_emoji = df["urdu_text"].apply(lambda t: len(describe(t)) > 0)
eda["share_with_emoji"] = float(has_emoji.mean())
eda["share_with_emoji_by_class"] = has_emoji.groupby(df["is_sarcastic"]).mean().round(3).to_dict()
eda["share_latin_tokens"] = float(df["urdu_text"].str.contains(r"[A-Za-z]{2,}").mean())
eda["share_hashtag"] = float(df["urdu_text"].str.contains(r"#|\w_\w").mean())
eda["share_mention"] = float(df["urdu_text"].str.contains("@").mean())
eda["share_url"] = float(df["urdu_text"].str.contains(r"http|www\.").mean())
eda["share_question"] = float(df["urdu_text"].str.contains(r"[?؟]").mean())
eda["share_quotes"] = float(df["urdu_text"].str.contains("``|''").mean())
prior = EmojiPrior().fit(train["urdu_text"], train["is_sarcastic"])
eda["emoji_logodds_top"] = prior.top(15)

fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
df["is_sarcastic"].value_counts().sort_index().plot.bar(ax=ax[0], color=["#4C72B0", "#DD8452"])
ax[0].set_title("class distribution (after cleaning)"); ax[0].set_xticklabels(["non-sarcastic (0)", "sarcastic (1)"], rotation=0)
for lab, c in [(0, "#4C72B0"), (1, "#DD8452")]:
    W[df["is_sarcastic"] == lab].plot.hist(bins=40, alpha=.6, ax=ax[1], color=c, label=f"class {lab}")
ax[1].set_title("words per post"); ax[1].legend()
top = (pd.Series([e.emoji for t in df["urdu_text"] for e in describe(t)]).value_counts().head(15))
top.plot.barh(ax=ax[2], color="#55A868"); ax[2].invert_yaxis(); ax[2].set_title("most frequent emojis")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "eda.png"), dpi=150); plt.close()

# emoji sarcasm-rate chart
rows = []
cnt = {}
for t, y in zip(train["urdu_text"], train["is_sarcastic"]):
    for e in {x.emoji for x in describe(t)}:
        cnt.setdefault(e, [0, 0])[int(y)] += 1
for e, (n0, n1) in cnt.items():
    if n0 + n1 >= 40:
        rows.append({"emoji": e, "n": n0 + n1, "sarcastic_rate": n1 / (n0 + n1)})
er = pd.DataFrame(rows).sort_values("sarcastic_rate")
er.to_csv(os.path.join(OUT, "emoji_sarcasm_rate.csv"), index=False, encoding="utf-8")
eda["emoji_sarcasm_rate_extremes"] = {"lowest": er.head(8).to_dict("records"), "highest": er.tail(8).to_dict("records")}

X_tr, y_tr = train["urdu_text"].tolist(), train["is_sarcastic"].values
X_va, y_va = val["urdu_text"].tolist(), val["is_sarcastic"].values
X_te, y_te = test["urdu_text"].tolist(), test["is_sarcastic"].values

# ----------------------------------------------------------------------------- ablations (validation set)
print("\n=== Ablations (LR, validation set) ===")
abl = run_ablations(X_tr, y_tr, X_va, y_va, clf="lr", C=2.0)
abl_df = pd.DataFrame(abl)
abl_df.to_csv(os.path.join(OUT, "ablations_val.csv"), index=False, encoding="utf-8")

# ----------------------------------------------------------------------------- classifier comparison (validation)
print("\n=== Classifier comparison on best feature set (validation) ===")
best_feat = "word+char+hand"
val_results, pipes = [], {}
for clf, Cs in [("lr", [0.5, 1, 2, 4, 8]), ("svm", [0.05, 0.1, 0.3, 1]), ("nb", [0.1, 0.3, 1.0])]:
    for C in Cs:
        name = f"{clf.upper()} C={C}"
        pipe = make_pipeline(best_feat, clf=clf, C=C, emoji_mode="both")
        r = fit_eval(pipe, X_tr, y_tr, X_va, y_va, name)
        r.update(clf=clf, C=C)
        val_results.append(r); pipes[name] = pipe
        print(f"{name:14s} val macro-F1 {r['macro_f1']:.4f}")
val_df = pd.DataFrame(val_results)
val_df.to_csv(os.path.join(OUT, "classical_val_search.csv"), index=False)

# ----------------------------------------------------------------------------- final test evaluation
print("\n=== Test set ===")
test_results = {}
final = []
for clf in ["lr", "svm", "nb"]:
    best = val_df[val_df.clf == clf].sort_values("macro_f1").iloc[-1]
    name = f"TF-IDF {clf.upper()} (word+char+hand)" if clf != "nb" else "TF-IDF ComplementNB (word+char)"
    pipe = pipes[best["model"]]
    pred = pipe.predict(X_te)
    r = evaluate(y_te, pred, name); r["C"] = float(best["C"]); r["val_macro_f1"] = float(best["macro_f1"])
    final.append(r)
    probs = predict_proba(pipe, X_te)
    pd.DataFrame({"urdu_text": X_te, "is_sarcastic": y_te, "pred": pred, "p_sarcastic": probs}).to_csv(
        os.path.join(OUT, f"pred_test_{clf}.csv"), index=False, encoding="utf-8")
    plot_confusion(r, os.path.join(FIG, f"cm_{clf}.png"))
    print(f"{name:40s} test macro-F1 {r['macro_f1']:.4f}  acc {r['accuracy']:.4f}")
    if clf == "lr":
        joblib.dump(pipe, os.path.join(OUT, "model_tfidf_lr.joblib"))
        tf = top_features(pipe, 30)
        json.dump(tf, open(os.path.join(OUT, "top_features_lr.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        err = error_table(test, pred, probs, k=25)
        err.to_csv(os.path.join(OUT, "errors_lr.csv"), index=False, encoding="utf-8")
        phenomena_summary(test, pred).to_csv(os.path.join(OUT, "phenomena_lr.csv"), index=False, encoding="utf-8")

# also: emoji-removed LR on test, to quantify the emoji contribution on the held-out set
pipe = make_pipeline("word+char", clf="lr", C=2.0, emoji_mode="remove")
r = fit_eval(pipe, X_tr, y_tr, X_te, y_te, "TF-IDF LR (word+char, emojis removed)")
final.append(r); print(f"{r['model']:40s} test macro-F1 {r['macro_f1']:.4f}")
pipe = make_pipeline("word+char", clf="lr", C=2.0, emoji_mode="both")
r = fit_eval(pipe, X_tr, y_tr, X_te, y_te, "TF-IDF LR (word+char, emoji semantics)")
final.append(r); print(f"{r['model']:40s} test macro-F1 {r['macro_f1']:.4f}")

results_table(final).to_csv(os.path.join(OUT, "results_classical_test.csv"), index=False)
json.dump({"eda": eda, "test": final, "ablations_val": abl, "elapsed_s": round(time.time() - t_start)},
          open(os.path.join(OUT, "classical_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(results_table(final).to_string())
print(f"done in {time.time() - t_start:.0f}s")
