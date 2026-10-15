# -*- coding: utf-8 -*-
"""
Fine-tuning a multilingual / Urdu-capable transformer for sarcasm detection.

Default: xlm-roberta-base (pretrained on CommonCrawl in 100 languages incl. Urdu).
Alternatives that also cover Urdu: google/muril-base-cased, bert-base-multilingual-cased,
distilbert-base-multilingual-cased (fastest), xlm-roberta-large (needs >12GB GPU).

Emoji handling for transformers
  * "keep": the SentencePiece vocabulary of XLM-R contains the most common emojis
    (😂, ❤) as single pieces, rarer ones are split into byte fallbacks.
  * "urdu": we replace each emoji by its Urdu gloss (😂 -> ہنسی کے آنسو) so that the
    *meaning* is expressed in a language the encoder understands.
  * "augment": keep the emoji AND append its Urdu meaning (😂 -> 😂 ہنسی کے آنسو) so the
    encoder sees both the symbol as used and what it means - all three are ablated.

A minimal explicit training loop is used (no HF Trainer) so that it runs the same
way on Colab GPU, a local GPU, or CPU, and so that mixed precision / early stopping
are transparent.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .emoji_semantics import emoji_to_tokens
from .evaluate import evaluate
from .preprocessing import normalize_social, normalize_urdu


def prepare_text(text: str, emoji_mode: str = "keep", normalize: bool = True) -> str:
    """Light preprocessing for transformers: the model is subword based, so we only
    unify Urdu code points and social-media artefacts; punctuation is left intact."""
    t = "" if text is None else str(text)
    if normalize:
        t = normalize_urdu(t)
        t = normalize_social(t, punct=False, laughter=False, elongation=True)
    if emoji_mode != "keep":
        t = emoji_to_tokens(t, emoji_mode)
    return t


def encode(tokenizer, texts: List[str], max_len: int = 96):
    enc = tokenizer(list(texts), padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


@torch.no_grad()
def predict(model, ids, mask, bs=64, device=None):
    device = device or next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, len(ids), bs):
        logits = model(input_ids=ids[i: i + bs].to(device), attention_mask=mask[i: i + bs].to(device)).logits
        out.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def fine_tune(model_name: str, train_texts, y_train, val_texts, y_val, *, emoji_mode="keep",
              max_len=96, epochs=3, bs=16, lr=2e-5, warmup=0.1, weight_decay=0.01, patience=1,
              fp16=None, seed=42, verbose=True, grad_accum=1, freeze_embeddings=True,
              lora_r=0, label_smoothing=0.0):
    """Returns (model, tokenizer, history, val_probs).

    freeze_embeddings: XLM-R's 250k x 768 word-embedding matrix holds 192M of its 278M
    parameters.  Freezing it (a) fits full fine-tuning of the 12 encoder layers into a
    6 GB GPU with AdamW, and (b) regularises training on a ~10k-example dataset where the
    embeddings would otherwise be updated from very few examples per sub-word."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp16 = (device.type == "cuda") if fp16 is None else fp16
    tok = AutoTokenizer.from_pretrained(model_name)
    extra_kw = {}
    if "deberta" in model_name.lower():
        # DeBERTa-v2/v3's disentangled attention overflows under fp16/bf16 autocast and the fused SDPA
        # path can yield NaN losses -> run it in fp32 with the reference (eager) attention implementation
        extra_kw["attn_implementation"] = "eager"
        fp16 = False
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, **extra_kw)
    if lora_r > 0:
        # LoRA (Hu et al. 2022): train small rank-r adapters on the attention/FFN projections and the
        # classifier head only -> xlm-roberta-large (560M) fits a 6 GB GPU; weights kept in bf16/fp16.
        from peft import LoraConfig, TaskType, get_peft_model
        cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.1,
                         target_modules=["query", "key", "value", "dense"])
        model = get_peft_model(model, cfg)
    elif freeze_embeddings:
        for p in model.base_model.embeddings.word_embeddings.parameters():
            p.requires_grad = False
    model.to(device)
    # bf16 autocast where supported (Ampere+): no overflow issues (DeBERTa) and no GradScaler needed
    use_bf16 = fp16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"trainable parameters: {n_train / 1e6:.1f}M / {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    tr_texts = [prepare_text(t, emoji_mode) for t in train_texts]
    va_texts = [prepare_text(t, emoji_mode) for t in val_texts]
    tr_ids, tr_mask = encode(tok, tr_texts, max_len)
    va_ids, va_mask = encode(tok, va_texts, max_len)
    y_tr = torch.as_tensor(np.asarray(y_train), dtype=torch.long)
    y_va = np.asarray(y_val)

    counts = np.bincount(y_tr.numpy(), minlength=2)
    w = torch.tensor(len(y_tr) / (2.0 * counts), dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=label_smoothing)

    no_decay = ("bias", "LayerNorm.weight")
    params = [{"params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)], "weight_decay": weight_decay},
              {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(params, lr=lr)
    steps_per_epoch = math.ceil(len(tr_ids) / bs / grad_accum)
    total = steps_per_epoch * epochs
    sched = get_linear_schedule_with_warmup(opt, int(warmup * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(fp16) and not use_bf16)

    best_f1, best_state, bad, hist, best_probs = -1.0, None, 0, [], None
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(tr_ids))
        t0, tot, n = time.time(), 0.0, 0
        opt.zero_grad()
        for step, i in enumerate(range(0, len(perm), bs)):
            j = perm[i: i + bs]
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=bool(fp16)):
                logits = model(input_ids=tr_ids[j].to(device), attention_mask=tr_mask[j].to(device)).logits
                loss = crit(logits.float(), y_tr[j].to(device)) / grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % grad_accum == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
            tot += loss.item() * grad_accum * len(j); n += len(j)
            if verbose and step % 100 == 0:
                print(f"  epoch {ep} step {step}/{len(perm) // bs}  loss {tot / max(n, 1):.4f}", flush=True)
        probs = predict(model, va_ids, va_mask, device=device)
        res = evaluate(y_va, (probs >= 0.5).astype(int), model_name)
        hist.append({"epoch": ep, "train_loss": tot / n, "val_macro_f1": res["macro_f1"], "val_acc": res["accuracy"],
                     "time_s": round(time.time() - t0)})
        if verbose:
            print(f"[{model_name}] epoch {ep}  loss {tot / n:.4f}  val macro-F1 {res['macro_f1']:.4f}  acc {res['accuracy']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if res["macro_f1"] > best_f1:
            best_f1, bad, best_probs = res["macro_f1"], 0, probs
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad > patience:
                break
    model.load_state_dict(best_state)
    return model, tok, hist, best_probs


def predict_texts(model, tok, texts, emoji_mode="keep", max_len=96, bs=64):
    ids, mask = encode(tok, [prepare_text(t, emoji_mode) for t in texts], max_len)
    return predict(model, ids, mask, bs=bs)
