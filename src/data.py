# -*- coding: utf-8 -*-
"""Loading, cleaning and leakage-free splitting of the Urdu sarcasm dataset."""
from __future__ import annotations

import os
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

TEXT, LABEL = "urdu_text", "is_sarcastic"


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    # the sheet export carries empty "Unnamed: n" columns -> keep only the two we need
    return df[[TEXT, LABEL]].copy()


def clean(df: pd.DataFrame, drop_conflicts: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    1. drop rows with missing text / label
    2. coerce label to int {0,1}
    3. drop exact duplicate (text,label) rows   -> otherwise the same post lands in
       train AND test and inflates every score (leakage)
    4. drop texts that appear with BOTH labels   -> annotation conflicts; a model
       cannot learn them and they would make the test set unanswerable
    5. drop texts that are empty after stripping
    Returns the cleaned frame and a dict of counts for the EDA report.
    """
    rep = {"rows_raw": len(df)}
    df = df.dropna(subset=[TEXT, LABEL]).copy()
    rep["rows_missing_dropped"] = rep["rows_raw"] - len(df)
    df[LABEL] = df[LABEL].astype(float).round().astype(int)
    df[TEXT] = df[TEXT].astype(str).str.strip()
    df = df[df[TEXT].str.len() > 0]
    before = len(df)
    df = df.drop_duplicates(subset=[TEXT, LABEL])
    rep["exact_duplicates_dropped"] = before - len(df)
    nun = df.groupby(TEXT)[LABEL].nunique()
    conflicts = set(nun[nun > 1].index)
    rep["conflicting_label_texts"] = len(conflicts)
    if drop_conflicts:
        before = len(df)
        df = df[~df[TEXT].isin(conflicts)]
        rep["conflicting_rows_dropped"] = before - len(df)
    rep["rows_clean"] = len(df)
    rep["class_counts"] = df[LABEL].value_counts().sort_index().to_dict()
    return df.reset_index(drop=True), rep


def split(df: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15, seed: int = 42):
    """Stratified train / validation / test split (default 70 / 15 / 15)."""
    train_val, test = train_test_split(df, test_size=test_size, stratify=df[LABEL], random_state=seed)
    rel_val = val_size / (1.0 - test_size)
    train, val = train_test_split(train_val, test_size=rel_val, stratify=train_val[LABEL], random_state=seed)
    assert not set(train[TEXT]) & set(test[TEXT]), "leakage: train/test share texts"
    assert not set(val[TEXT]) & set(test[TEXT]), "leakage: val/test share texts"
    return (train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True))


def load_and_split(path: str, seed: int = 42, drop_conflicts: bool = True):
    df, rep = clean(load_raw(path), drop_conflicts=drop_conflicts)
    train, val, test = split(df, seed=seed)
    rep["split"] = {"train": len(train), "val": len(val), "test": len(test)}
    return df, train, val, test, rep


def save_splits(train, val, test, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for name, d in [("train", train), ("val", val), ("test", test)]:
        d.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False, encoding="utf-8")
