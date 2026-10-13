# -*- coding: utf-8 -*-
"""
Predict + explain: the sarcasm label **together with the meaning of every emoji** in the post
and the linguistic cues that fired.

    >>> ex = SarcasmExplainer.load()            # TF-IDF LR saved by scripts/run_classical.py
    >>> ex.explain("واہ خان صاحب کیا بات ہے 😂🙄")
    {'label': 'sarcastic', 'p_sarcastic': 0.91,
     'emojis': [{'emoji': '😂', 'meaning_en': 'face with tears of joy', 'meaning_ur': 'ہنسی کے آنسو', ...}, ...],
     'emoji_summary_ur': '😂 = ہنسی کے آنسو، 🙄 = آنکھیں گھمانا / بیزاری',
     'cues': ['ironic-praise marker (واہ, کیا بات ہے, خان صاحب)', 'irony-cue emoji (😂, 🙄)', 'positive words + ironic emoji'],
     'reading': 'Praise words combined with laughing/eye-rolling emojis -> the praise is not meant literally.'}
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

import numpy as np

from .emoji_semantics import describe
from .features import (IDIOMS, IDIOM_RE, NEGATION_WORDS, POSITIVE_WORDS, NEGATIVE_WORDS,
                       SARCASM_MARKER_RE)
from .preprocessing import normalize_urdu

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SarcasmExplainer:
    def __init__(self, predict_proba, name="model"):
        """predict_proba: callable(list[str]) -> np.array of P(sarcastic)."""
        self.predict_proba = predict_proba
        self.name = name

    # ------------------------------------------------------------------ loaders
    @classmethod
    def load(cls, path: Optional[str] = None):
        """Load the saved TF-IDF logistic-regression pipeline."""
        import joblib
        from .baselines import predict_proba as _pp
        path = path or os.path.join(ROOT, "outputs", "model_tfidf_lr.joblib")
        pipe = joblib.load(path)
        return cls(lambda texts: _pp(pipe, list(texts)), name="TF-IDF LR")

    @classmethod
    def load_transformer(cls, path: str, emoji_mode: str = "augment", max_len: int = 96):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from .transformer import predict_texts
        import torch
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        return cls(lambda texts: predict_texts(model, tok, list(texts), emoji_mode=emoji_mode, max_len=max_len),
                   name=os.path.basename(path))

    # ------------------------------------------------------------------ explain
    def explain(self, text: str, threshold: float = 0.5) -> Dict:
        p = float(self.predict_proba([text])[0])
        emojis = describe(text)
        t = normalize_urdu(text)
        words = re.findall(r"[؀-ۿ]+", t)
        cues: List[str] = []
        markers = SARCASM_MARKER_RE.findall(t)
        if markers:
            cues.append("ironic-praise marker (" + ", ".join(dict.fromkeys(markers)) + ")")
        idioms = IDIOM_RE.findall(t)
        for i in dict.fromkeys(idioms):
            cues.append(f"idiom: {i} = {IDIOMS.get(i, '')}")
        cue_emojis = [e.emoji for e in emojis if e.cue]
        if cue_emojis:
            cues.append("irony-cue emoji (" + ", ".join(dict.fromkeys(cue_emojis)) + ")")
        pos = [w for w in words if w in POSITIVE_WORDS]
        neg = [w for w in words if w in NEGATIVE_WORDS]
        if (pos or markers) and (cue_emojis or any(e.polarity < 0 for e in emojis)):
            cues.append("positive words + ironic/negative emoji (polarity contrast)")
        if neg and any(e.polarity > 0 and e.category in ("LOVE", "PRAISE", "FLOWER") for e in emojis):
            cues.append("negative words + affectionate emoji (polarity contrast)")
        if any(w in NEGATION_WORDS for w in words):
            cues.append("negation")
        if re.search(r"[?؟]{2,}|\.{3,}|۔{3,}", text):
            cues.append("rhetorical punctuation")
        if re.search(r"(ہا|ھا){3,}|lol|haha", text, re.I):
            cues.append("textual laughter")
        if len(set(e.emoji for e in emojis)) < len(emojis):
            cues.append("repeated emoji (intensity)")
        if re.search(r"[A-Za-z]{2,}", text):
            cues.append("code-mixing")

        emo_rows = [dict(emoji=e.emoji, meaning_en=e.en, meaning_ur=e.ur, polarity=e.polarity,
                         category=e.category, irony_cue=e.cue, position=e.position) for e in emojis]
        uniq = list(dict.fromkeys((e.emoji, e.ur, e.en) for e in emojis))
        summary_ur = "، ".join(f"{em} = {u}" for em, u, _ in uniq) if uniq else "کوئی ایموجی نہیں"
        summary_en = "; ".join(f"{em} = {en}" for em, _, en in uniq) if uniq else "no emoji"
        label = "sarcastic" if p >= threshold else "non-sarcastic"
        return dict(text=text, label=label, label_ur="طنزیہ" if label == "sarcastic" else "غیر طنزیہ",
                    p_sarcastic=round(p, 3), model=self.name, emojis=emo_rows,
                    emoji_summary_ur=summary_ur, emoji_summary_en=summary_en, cues=cues,
                    reading=self._reading(label, emojis, cues))

    @staticmethod
    def _reading(label, emojis, cues):
        if not emojis and not cues:
            return "No emoji or lexical cue; decision rests on the wording alone."
        parts = []
        cats = {e.category for e in emojis}
        if label == "sarcastic":
            if cats & {"LAUGH", "TEASE", "SMIRK", "THINK"}:
                parts.append("laughing / teasing / smirking emojis signal that the statement is not meant literally")
            if any("praise" in c for c in cues):
                parts.append("the praise words are a vehicle for mockery")
            if any("contrast" in c for c in cues):
                parts.append("the words and the emoji disagree in polarity")
        else:
            if cats & {"LOVE", "FLOWER", "PRAYER", "SAD"}:
                parts.append("hearts / flowers / prayer / sadness emojis belong to the sincere (poetry, greeting, grief) register")
            if cats & {"LAUGH"} and not cues:
                parts.append("laughter emoji with no ironic wording reads as plain humour")
        return "; ".join(parts) + "." if parts else "Decision driven by the wording; emojis are neutral here."

    def explain_many(self, texts: List[str], threshold: float = 0.5):
        import pandas as pd
        rows = [self.explain(t, threshold) for t in texts]
        return pd.DataFrame([{k: (", ".join(v) if isinstance(v, list) and v and isinstance(v[0], str) else v)
                              for k, v in r.items() if k != "emojis"} for r in rows])


def format_explanation(r: Dict) -> str:
    lines = [f"TEXT   : {r['text']}",
             f"LABEL  : {r['label']} / {r['label_ur']}   (P(sarcastic) = {r['p_sarcastic']:.2f}, {r['model']})",
             "EMOJIS :" + ("" if r["emojis"] else " none")]
    for e in r["emojis"]:
        lines.append(f"   {e['emoji']}  {e['meaning_en']:32s} {e['meaning_ur']:24s} [{e['category']}, polarity {e['polarity']:+d}{', irony cue' if e['irony_cue'] else ''}, {e['position']}]")
    lines.append("CUES   : " + ("; ".join(r["cues"]) if r["cues"] else "none"))
    lines.append("READING: " + r["reading"])
    return "\n".join(lines)
