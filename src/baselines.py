# -*- coding: utf-8 -*-
"""
Classical TF-IDF baselines and the ablation runner.

Feature groups
  word  : word uni+bi-grams on the preprocessed token string
  char  : character 2-5-grams inside word boundaries (robust to Urdu spelling
          variation, to morphological suffixes and to the lack of standard spacing)
  hand  : dense hand-crafted linguistic + emoji features (features.py)

Classifiers
  LogisticRegression, LinearSVC (calibrated for probabilities) and ComplementNB.
"""
from __future__ import annotations

import time
from functools import partial
from typing import Dict, List, Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .evaluate import evaluate
from .features import HandcraftedFeatures
from .preprocessing import preprocess


def _pre(normalize=True, social=True, emoji_mode="both"):
    return partial(preprocess, normalize=normalize, social=social, emoji_mode=emoji_mode)


def make_features(kind: str = "word+char+hand", *, normalize=True, social=True,
                  emoji_mode="both", word_ngram=(1, 2), char_ngram=(2, 5),
                  min_df=2, max_char_features=300_000, use_prior=True):
    """Build the sklearn feature pipeline described by `kind` (e.g. 'word', 'char', 'word+char+hand')."""
    pre = _pre(normalize, social, emoji_mode)
    parts = []
    if "word" in kind:
        parts.append(("word", TfidfVectorizer(preprocessor=pre, tokenizer=str.split, token_pattern=None,
                                              ngram_range=word_ngram, min_df=min_df, sublinear_tf=True,
                                              lowercase=False)))
    if "char" in kind:
        parts.append(("char", TfidfVectorizer(preprocessor=pre, analyzer="char_wb", ngram_range=char_ngram,
                                              min_df=min_df, sublinear_tf=True, lowercase=False,
                                              max_features=max_char_features)))
    if "hand" in kind:
        parts.append(("hand", Pipeline([("feat", HandcraftedFeatures(use_prior=use_prior)),
                                        ("scale", StandardScaler())])))
    if not parts:
        raise ValueError(kind)
    return FeatureUnion(parts) if len(parts) > 1 else parts[0][1]


def make_classifier(name: str = "lr", C: float = 1.0, class_weight="balanced"):
    if name == "lr":
        return LogisticRegression(C=C, max_iter=3000, class_weight=class_weight, solver="liblinear")
    if name == "svm":
        return CalibratedClassifierCV(LinearSVC(C=C, class_weight=class_weight), cv=3)
    if name == "nb":
        return ComplementNB(alpha=C)
    raise ValueError(name)


def make_pipeline(features="word+char+hand", clf="lr", C=1.0, **feat_kw) -> Pipeline:
    if clf == "nb" and "hand" in features:
        # NB needs non-negative inputs; the scaled dense block is signed -> drop it
        features = features.replace("+hand", "")
    return Pipeline([("features", make_features(features, **feat_kw)),
                     ("clf", make_classifier(clf, C=C))])


def fit_eval(pipe: Pipeline, X_train, y_train, X_eval, y_eval, name: str) -> Dict:
    t0 = time.time()
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_eval)
    res = evaluate(y_eval, pred, name)
    res["train_time_s"] = round(time.time() - t0, 1)
    try:
        res["n_features"] = int(pipe.named_steps["features"].transform(X_eval[:2]).shape[1])
    except Exception:
        pass
    return res


def predict_proba(pipe: Pipeline, X):
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    s = pipe.decision_function(X)
    return 1 / (1 + np.exp(-s))


# --------------------------------------------------------------------------- #
# ablation configurations                                                     #
# --------------------------------------------------------------------------- #
ABLATIONS: List[dict] = [
    # ---- A. what to do with emojis (word+char features, LR) ---------------
    dict(name="A1 emojis removed",          features="word+char", emoji_mode="remove"),
    dict(name="A2 emojis kept raw",         features="word+char", emoji_mode="keep"),
    dict(name="A3 emojis -> meaning token", features="word+char", emoji_mode="demojize"),
    dict(name="A4 emojis -> category",      features="word+char", emoji_mode="category"),
    dict(name="A5 emojis -> Urdu gloss",    features="word+char", emoji_mode="urdu"),
    dict(name="A6 emojis -> meaning+category+polarity", features="word+char", emoji_mode="both"),
    dict(name="A7 emoji symbol + Urdu meaning (augment)", features="word+char", emoji_mode="augment"),
    # ---- B. feature type ---------------------------------------------------
    dict(name="B1 word n-grams only",       features="word",      emoji_mode="both"),
    dict(name="B2 char n-grams only",       features="char",      emoji_mode="both"),
    dict(name="B3 word+char",               features="word+char", emoji_mode="both"),
    dict(name="B4 hand-crafted only",       features="hand",      emoji_mode="both"),
    dict(name="B5 word+char+hand-crafted",  features="word+char+hand", emoji_mode="both"),
    # ---- C. normalisation --------------------------------------------------
    dict(name="C1 raw text (no normalisation)", features="word+char", normalize=False, social=False, emoji_mode="keep"),
    dict(name="C2 Urdu char normalisation only", features="word+char", normalize=True, social=False, emoji_mode="keep"),
    dict(name="C3 Urdu + social normalisation",  features="word+char", normalize=True, social=True, emoji_mode="keep"),
]


def run_ablations(X_train, y_train, X_val, y_val, clf="lr", C=1.0, configs: Optional[List[dict]] = None,
                  verbose=True) -> List[Dict]:
    out = []
    for cfg in (configs or ABLATIONS):
        cfg = dict(cfg)
        name = cfg.pop("name")
        feats = cfg.pop("features")
        pipe = make_pipeline(feats, clf=clf, C=C, **cfg)
        res = fit_eval(pipe, X_train, y_train, X_val, y_val, name)
        res.update(dict(features=feats, **cfg))
        out.append(res)
        if verbose:
            print(f"{name:45s} macro-F1 {res['macro_f1']:.4f}  acc {res['accuracy']:.4f}  ({res['train_time_s']}s)")
    return out


def top_features(pipe: Pipeline, n: int = 25):
    """Most sarcastic / least sarcastic features of a linear model (LR only)."""
    fu = pipe.named_steps["features"]
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return None
    names = []
    for nm, tr in fu.transformer_list:
        if nm == "hand":
            names += ["hand:" + f for f in tr.named_steps["feat"].get_feature_names_out()]
        else:
            names += [nm + ":" + f for f in tr.get_feature_names_out()]
    coef = clf.coef_.ravel()
    order = np.argsort(coef)
    return {"sarcastic": [(names[i], round(float(coef[i]), 3)) for i in order[::-1][:n]],
            "non_sarcastic": [(names[i], round(float(coef[i]), 3)) for i in order[:n]]}
