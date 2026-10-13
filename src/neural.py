# -*- coding: utf-8 -*-
"""
Neural sarcasm classifiers in PyTorch.

  TextCNN      : multi-width 1-D convolutions over word+emoji embeddings (n-gram detectors)
  BiLSTMAttn   : bidirectional LSTM with additive attention pooling (long-range
                 context, e.g. a praise phrase at the start and 😂 at the end)
Both accept an optional dense vector of hand-crafted features ("hybrid" variants),
which is concatenated with the pooled text representation before the classifier.

The tokenizer keeps each emoji as ONE token, so the model learns an embedding
for 😂 exactly like for a word, and emoji *sequences* (😂😂🙄) are visible to the CNN.
"""
from __future__ import annotations

import math
import random
import time
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .evaluate import evaluate
from .preprocessing import preprocess

PAD, UNK = "<pad>", "<unk>"


def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Vocab:
    def __init__(self, min_freq: int = 2, max_size: int = 50_000, emoji_mode: str = "keep"):
        self.min_freq, self.max_size, self.emoji_mode = min_freq, max_size, emoji_mode
        self.itos: List[str] = [PAD, UNK]
        self.stoi: Dict[str, int] = {PAD: 0, UNK: 1}

    def tokenize(self, text: str) -> List[str]:
        return preprocess(text, emoji_mode=self.emoji_mode).split()

    def fit(self, texts):
        c = Counter(tok for t in texts for tok in self.tokenize(t))
        for tok, f in c.most_common(self.max_size):
            if f < self.min_freq:
                break
            self.stoi[tok] = len(self.itos)
            self.itos.append(tok)
        return self

    def encode(self, text: str, max_len: int = 64) -> List[int]:
        ids = [self.stoi.get(t, 1) for t in self.tokenize(text)][:max_len]
        return ids + [0] * (max_len - len(ids))

    def __len__(self):
        return len(self.itos)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, emb_dim=200, n_filters=128, kernel_sizes=(2, 3, 4, 5),
                 dropout=0.5, n_dense=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, n_filters, k, padding=k // 2) for k in kernel_sizes])
        self.drop = nn.Dropout(dropout)
        self.n_dense = n_dense
        if n_dense:
            self.dense_proj = nn.Sequential(nn.Linear(n_dense, 64), nn.ReLU(), nn.Dropout(dropout))
        self.out = nn.Linear(n_filters * len(kernel_sizes) + (64 if n_dense else 0), 2)

    def forward(self, x, dense=None):
        e = self.drop(self.emb(x)).transpose(1, 2)           # B x E x T
        mask = (x != 0).unsqueeze(1)                          # B x 1 x T
        pooled = []
        for conv in self.convs:
            h = F.relu(conv(e))[:, :, : x.size(1)]
            h = h.masked_fill(~mask, -1e4)
            pooled.append(h.max(dim=2).values)
        h = self.drop(torch.cat(pooled, dim=1))
        if self.n_dense:
            h = torch.cat([h, self.dense_proj(dense)], dim=1)
        return self.out(h)


class BiLSTMAttn(nn.Module):
    def __init__(self, vocab_size, emb_dim=200, hidden=128, n_layers=1, dropout=0.5, n_dense=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=n_layers, batch_first=True, bidirectional=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.att = nn.Linear(2 * hidden, 1)
        self.drop = nn.Dropout(dropout)
        self.n_dense = n_dense
        if n_dense:
            self.dense_proj = nn.Sequential(nn.Linear(n_dense, 64), nn.ReLU(), nn.Dropout(dropout))
        self.out = nn.Linear(4 * hidden + (64 if n_dense else 0), 2)

    def forward(self, x, dense=None, return_attention=False):
        mask = x != 0
        lengths = mask.sum(1).clamp(min=1).cpu()
        e = self.drop(self.emb(x))
        packed = nn.utils.rnn.pack_padded_sequence(e, lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=x.size(1))  # B x T x 2H
        scores = self.att(h).squeeze(-1).masked_fill(~mask, -1e4)
        alpha = torch.softmax(scores, dim=1)                   # B x T
        att_pool = (alpha.unsqueeze(-1) * h).sum(1)
        max_pool = h.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values
        z = self.drop(torch.cat([att_pool, max_pool], dim=1))
        if self.n_dense:
            z = torch.cat([z, self.dense_proj(dense)], dim=1)
        logits = self.out(z)
        return (logits, alpha) if return_attention else logits


# --------------------------------------------------------------------------- #
# training                                                                    #
# --------------------------------------------------------------------------- #
def _batches(X, D, y, bs, shuffle):
    idx = np.arange(len(X))
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, len(idx), bs):
        j = idx[i: i + bs]
        yield X[j], (D[j] if D is not None else None), (y[j] if y is not None else None)


@torch.no_grad()
def predict(model, X, D=None, bs=256, device=None):
    device = device or get_device()
    model.eval()
    probs = []
    for xb, db, _ in _batches(X, D, None, bs, False):
        xb = torch.as_tensor(xb, device=device)
        db = torch.as_tensor(db, device=device) if db is not None else None
        probs.append(torch.softmax(model(xb, db), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def train_model(model, X_tr, y_tr, X_va, y_va, D_tr=None, D_va=None, *, epochs=15, bs=64, lr=1e-3,
                weight_decay=1e-5, patience=3, class_weight=True, device=None, verbose=True, name="model"):
    """Adam + early stopping on validation macro-F1.  Returns (best_model, history)."""
    device = device or get_device()
    model.to(device)
    if class_weight:
        counts = np.bincount(y_tr, minlength=2)
        w = torch.tensor(len(y_tr) / (2.0 * counts), dtype=torch.float32, device=device)
    else:
        w = None
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)
    best_f1, best_state, bad, hist = -1, None, 0, []
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for xb, db, yb in _batches(X_tr, D_tr, y_tr, bs, True):
            xb = torch.as_tensor(xb, device=device)
            yb = torch.as_tensor(yb, device=device)
            db = torch.as_tensor(db, device=device) if db is not None else None
            opt.zero_grad()
            loss = crit(model(xb, db), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(xb); n += len(xb)
        p = predict(model, X_va, D_va, device=device)
        res = evaluate(y_va, (p >= 0.5).astype(int), name)
        sched.step(res["macro_f1"])
        hist.append({"epoch": ep, "train_loss": tot / n, "val_macro_f1": res["macro_f1"], "val_acc": res["accuracy"]})
        if verbose:
            print(f"[{name}] epoch {ep:2d}  loss {tot / n:.4f}  val macro-F1 {res['macro_f1']:.4f}  acc {res['accuracy']:.4f}  ({time.time() - t0:.0f}s)")
        if res["macro_f1"] > best_f1:
            best_f1, bad = res["macro_f1"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.to(device)
    return model, hist


def best_threshold(y_true, probs):
    """Pick the probability threshold that maximises macro-F1 on the VALIDATION set."""
    best_t, best_f = 0.5, -1
    for t in np.arange(0.2, 0.81, 0.02):
        f = evaluate(y_true, (probs >= t).astype(int))["macro_f1"]
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t, best_f
