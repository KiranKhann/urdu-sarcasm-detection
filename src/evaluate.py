# -*- coding: utf-8 -*-
"""Metrics, confusion matrices, results tables and linguistic error analysis."""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from .emoji_semantics import describe, explain
from .features import IDIOM_RE, NEGATION_WORDS, SARCASM_MARKER_RE
from .preprocessing import normalize_urdu

LABEL_NAMES = {0: "non-sarcastic", 1: "sarcastic"}


def evaluate(y_true, y_pred, name: str = "") -> Dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_0": p[0], "recall_0": r[0], "f1_0": f[0],
        "precision_1": p[1], "recall_1": r[1], "f1_1": f[1],
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "support_0": int(s[0]), "support_1": int(s[1]),
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    }


def results_table(results: List[Dict]) -> pd.DataFrame:
    cols = ["model", "accuracy", "precision_0", "recall_0", "f1_0",
            "precision_1", "recall_1", "f1_1", "macro_f1"]
    df = pd.DataFrame(results)[cols].copy()
    return df.sort_values("macro_f1", ascending=False).reset_index(drop=True)


def plot_confusion(res: Dict, path: Optional[str] = None, ax=None, title: Optional[str] = None):
    import matplotlib.pyplot as plt
    cm = np.array([[res["tn"], res["fp"]], [res["fn"], res["tp"]]])
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / cm[i].sum():.0%})", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred 0", "pred 1"]); ax.set_yticklabels(["true 0", "true 1"])
    ax.set_title(title or f"{res['model']}\nmacro-F1 {res['macro_f1']:.3f}", fontsize=9)
    if own:
        plt.tight_layout()
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path, dpi=150)
        plt.close(fig)
    return ax


# --------------------------------------------------------------------------- #
# error analysis: tag each misclassified post with linguistic phenomena       #
# --------------------------------------------------------------------------- #
_LATIN = re.compile(r"[A-Za-z]{2,}")
_SPELL = re.compile(r"ھے|ھیں|ھو\b|ہوئ\b|کوئ\b|ۓ|ٸ|ٶ|ك|ي")
_PUNJABI = re.compile(r"(?<![؀-ۿ])(تسی|ساڈی|ساڈا|تہانوں|توانوں|مینوں|نوں|وچ|ایتھے|اوتھے|کیتی|کیتا|ہوندا|ہوندی|دا|دی|دے|نئیں|ہیگا|کی گل|چھڈو|لگدا)(?![؀-ۿ])")
_QUESTION = re.compile(r"[?؟]")
_QUOTE = re.compile("``|''|\"|“|”")
_LAUGH = re.compile(r"(?:ہا|ھا|ہہ){3,}|lol", re.I)


def phenomena(text: str) -> List[str]:
    """Automatic (heuristic) tags; the notebook adds human commentary on top."""
    t = normalize_urdu(text)
    tags = []
    emo = describe(text)
    if emo:
        tags.append("emoji")
        if any(e.cue for e in emo):
            tags.append("irony-cue emoji")
        pols = {e.polarity for e in emo}
        if 1 in pols and -1 in pols:
            tags.append("mixed-polarity emojis")
        if all(e.category == "LAUGH" for e in emo):
            tags.append("laughter-only emoji")
    if any(w in NEGATION_WORDS for w in re.findall(r"[؀-ۿ]+", t)):
        tags.append("negation")
    if IDIOM_RE.search(t):
        tags.append("idiom/phrase")
    if SARCASM_MARKER_RE.search(t):
        tags.append("ironic-praise marker")
    if _LATIN.search(t):
        tags.append("code-mixing (Latin)")
    if _PUNJABI.search(t):
        tags.append("Punjabi")
    if _SPELL.search(text):
        tags.append("spelling variant")
    if _QUESTION.search(t):
        tags.append("rhetorical question")
    if _QUOTE.search(t):
        tags.append("quoted speech")
    if _LAUGH.search(t):
        tags.append("textual laughter")
    if len(t.split()) <= 4:
        tags.append("very short / context-dependent")
    if re.search(r"(عمران|خان|بلاول|نواز|زرداری|مریم|فوج|آرمی|نوٹس|حکومت|وزیر|پی ٹی آئی|پیپلز|لیگ|کامران|حامد)", t):
        tags.append("political context")
    return tags


def error_table(test_df: pd.DataFrame, y_pred, proba=None, text_col="urdu_text",
                label_col="is_sarcastic", k: int = 20, seed: int = 0) -> pd.DataFrame:
    df = test_df[[text_col, label_col]].copy()
    df["pred"] = np.asarray(y_pred)
    if proba is not None:
        df["p_sarcastic"] = np.asarray(proba)
    err = df[df[label_col] != df["pred"]].copy()
    if proba is not None:   # most confident mistakes are the most informative
        err["conf"] = (err["p_sarcastic"] - 0.5).abs()
        err = err.sort_values("conf", ascending=False)
    else:
        err = err.sample(frac=1, random_state=seed)
    err = err.head(k).copy()
    err["error_type"] = np.where(err[label_col] == 1, "false negative (missed sarcasm)",
                                 "false positive (literal read as sarcasm)")
    err["emoji_meaning"] = err[text_col].apply(explain)
    err["phenomena"] = err[text_col].apply(lambda s: ", ".join(phenomena(s)))
    return err.reset_index(drop=True)


def phenomena_summary(test_df: pd.DataFrame, y_pred, text_col="urdu_text", label_col="is_sarcastic"):
    """Error rate per linguistic phenomenon across the whole test set."""
    rows = []
    tags = test_df[text_col].apply(phenomena)
    wrong = (test_df[label_col].values != np.asarray(y_pred))
    for tag in sorted({t for ts in tags for t in ts}):
        mask = tags.apply(lambda ts: tag in ts).values
        rows.append({"phenomenon": tag, "n_posts": int(mask.sum()),
                     "error_rate": float(wrong[mask].mean()) if mask.any() else np.nan})
    return pd.DataFrame(rows).sort_values("error_rate", ascending=False).reset_index(drop=True)
