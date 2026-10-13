# -*- coding: utf-8 -*-
"""
Urdu-aware normalisation and tokenisation for social-media text.

Design principles (justified in the notebook):
  * Emojis are extracted BEFORE any character normalisation, because the
    zero-width joiner (U+200D) and variation selector (U+FE0F) that we strip
    from Urdu text are *part of* emoji sequences (🤦‍♂️).
  * Normalisation is conservative: we unify Arabic/Urdu code-point variants
    (ك/ک, ي/ی, ة/ہ, أ/ا ...) and strip diacritics, because the same word is
    typed many ways on phones, but we do NOT remove punctuation, repetition,
    hashtags, mentions or Roman/English tokens by default - each of those is a
    candidate sarcasm cue and is tested in the ablation study.
  * Punjabi, Roman-Urdu and English tokens are kept (code-mixing is frequent:
    ~15% of posts contain Latin-script tokens).
"""
from __future__ import annotations

import re
import unicodedata
from typing import List

from .emoji_semantics import emoji_to_tokens, extract

# --------------------------------------------------------------------------- #
# character-level normalisation tables                                        #
# --------------------------------------------------------------------------- #
CHAR_MAP = {
    "ك": "ک", "ي": "ی", "ى": "ی", "ۓ": "ئے",
    "ة": "ہ", "ۃ": "ہ", "ە": "ہ", "ۀ": "ہ", "ۂ": "ہ",
    "أ": "ا", "إ": "ا", "ٱ": "ا", "ٲ": "ا", "ٳ": "ا",
    "ٶ": "ؤ", "ٸ": "ئ",          # Sindhi/Pashto forms seen in the data (ناٸیٹ, لٸے)
    "‌": "", "‍": "",  # ZWNJ, ZWJ (safe: emojis are protected first)
    "ـ": "",                # tatweel
    "٫": ".", "٬": ",",
}
# Urdu & Arabic-Indic digits -> ASCII
for _i, _d in enumerate("۰۱۲۳۴۵۶۷۸۹"):
    CHAR_MAP[_d] = str(_i)
for _i, _d in enumerate("٠١٢٣٤٥٦٧٨٩"):
    CHAR_MAP[_d] = str(_i)

_CHAR_RE = re.compile("|".join(map(re.escape, CHAR_MAP)))
# harakat / Quranic annotation marks
_DIACRITICS = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭ]")

# Common keyboard spelling variants: ھ (do-chashmi he) typed for ہ in function words
WORD_FIXES = {
    "ھے": "ہے", "ھیں": "ہیں", "ھو": "ہو", "ھوں": "ہوں", "ھی": "ہی", "ھم": "ہم",
    "ھاں": "ہاں", "ھر": "ہر", "ھوا": "ہوا", "ھوگا": "ہوگا", "ھوگی": "ہوگی", "ھوئی": "ہوئی",
    "ھوئے": "ہوئے", "ھمارا": "ہمارا", "ھماری": "ہماری", "ھمارے": "ہمارے", "ھوتا": "ہوتا",
    "ھوتی": "ہوتی", "ھوتے": "ہوتے", "ھونا": "ہونا", "ھونے": "ہونے",
    "کیلئے": "کے لیے", "کیلیے": "کے لیے", "لئے": "لیے",
    "ہوئ": "ہوئی", "کوئ": "کوئی", "گئ": "گئی", "نئ": "نئی",
}
_WORD_FIX_RE = re.compile(
    r"(?<![؀-ۿ])(" + "|".join(map(re.escape, sorted(WORD_FIXES, key=len, reverse=True))) + r")(?![؀-ۿ])"
)

# Urdu punctuation -> ASCII equivalents (kept as tokens)
PUNCT_MAP = {"۔": ".", "،": ",", "؟": "?", "؛": ";", "٪": "%",
             "“": '"', "”": '"', "‘": "'", "’": "'", "…": "...",
             "``": '"', "''": '"'}

_URL = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(?:com|pk|net|org)\S*)", re.I)
_MENTION = re.compile(r"@\w+")
_HASHTAG = re.compile(r"#(\w+)")
# a hashtag that lost its '#' in scraping shows up as words_joined_by_underscores
_UNDERSCORE_TAG = re.compile(r"(?<!\S)([\w؀-ۿ]+(?:_[\w؀-ۿ]+)+)(?!\S)")
_LAUGH = re.compile(r"\b(?:(?:ہا|ھا|اہ|ہہ|ھہ){2,}[ہھا]*|(?:ha){2,}h?|(?:he){2,}h?|lol+|lmao)\b", re.I)
_ELONG = re.compile(r"(.)\1{2,}")             # کدوووووو -> کدوو
_PUNCT_REPEAT = re.compile(r"([!?.])\1{2,}")  # ؟؟؟؟؟ -> ???
_WS = re.compile(r"\s+")


def normalize_urdu(text: str, *, fix_words: bool = True, strip_diacritics: bool = True) -> str:
    """Character-level normalisation.  Emojis are protected (never touched)."""
    text = unicodedata.normalize("NFC", str(text))
    spans = extract(text)
    pieces, last = [], 0
    for s in spans:
        pieces.append(("t", text[last: s["match_start"]]))
        pieces.append(("e", s["emoji"]))
        last = s["match_end"]
    pieces.append(("t", text[last:]))
    out = []
    for kind, chunk in pieces:
        if kind == "e":
            out.append(chunk)
            continue
        chunk = _CHAR_RE.sub(lambda m: CHAR_MAP[m.group(0)], chunk)
        if strip_diacritics:
            chunk = _DIACRITICS.sub("", chunk)
        if fix_words:
            chunk = _WORD_FIX_RE.sub(lambda m: WORD_FIXES[m.group(1)], chunk)
        out.append(chunk)
    return "".join(out)


def normalize_social(text: str, *, urls=True, mentions=True, hashtags=True,
                     laughter=True, elongation=True, punct=True) -> str:
    """Social-media specific normalisation; every step is switchable for ablation."""
    t = str(text)
    if urls:
        t = _URL.sub(" URL ", t)
    if mentions:
        t = _MENTION.sub(" @USER ", t)
    if hashtags:
        t = _HASHTAG.sub(lambda m: " HASHTAG " + m.group(1).replace("_", " ") + " ", t)
        t = _UNDERSCORE_TAG.sub(lambda m: " HASHTAG " + m.group(1).replace("_", " ") + " ", t)
    if laughter:
        t = _LAUGH.sub(" LAUGH ", t)
    if elongation:
        t = _ELONG.sub(r"\1\1", t)
    if punct:
        for k, v in PUNCT_MAP.items():
            t = t.replace(k, v)
        t = _PUNCT_REPEAT.sub(r"\1\1\1", t)
        # space out punctuation so it becomes a token (keeps ... and ??? together)
        t = re.sub(r"(\.{2,}|\?{2,}|!{2,}|[.,?!;:\"'()])", r" \1 ", t)
    return _WS.sub(" ", t).strip()


# token = emoji | Urdu/Arabic word | marker | Latin word | number | punctuation run
_TOKEN_RE = re.compile(
    r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]+"
    r"|EMO(?:CAT|POL)?_\w+|EMOCUE|EMO_REPEAT|HASHTAG|LAUGH|URL|@USER"
    r"|[A-Za-z]+(?:'[A-Za-z]+)?"
    r"|\d+(?:[.,]\d+)?"
    r"|\.{2,}|\?{2,}|!{2,}|[^\s\w]"
)


def tokenize(text: str) -> List[str]:
    """Regex tokenizer that keeps emojis as single tokens."""
    toks, last = [], 0
    for s in extract(text):
        toks += _TOKEN_RE.findall(text[last: s["match_start"]])
        toks.append(s["emoji"])
        last = s["match_end"]
    toks += _TOKEN_RE.findall(text[last:])
    return toks


_MARKERS = ("EMO", "HASHTAG", "LAUGH", "URL", "USER")


def preprocess(text: str, *, normalize: bool = True, social: bool = True,
               emoji_mode: str = "both", lowercase: bool = True) -> str:
    """
    Full pipeline -> a whitespace-separated token string ready for TF-IDF / tokenizers.

    emoji_mode is one of: keep | remove | demojize | category | both | urdu
    (see emoji_semantics.emoji_to_tokens).
    """
    t = "" if text is None else str(text)
    if normalize:
        t = normalize_urdu(t)
    if social:
        t = normalize_social(t)
    t = emoji_to_tokens(t, emoji_mode)
    if lowercase:   # only Latin letters are affected; Urdu has no case
        t = re.sub(r"[A-Za-z_]+",
                   lambda m: m.group(0) if m.group(0).startswith(_MARKERS) else m.group(0).lower(), t)
    return " ".join(tokenize(t))


# variants used in the ablation study
PREPROCESS_VARIANTS = {
    "raw":                 dict(normalize=False, social=False, emoji_mode="keep", lowercase=False),
    "norm_emoji_removed":  dict(normalize=True,  social=True,  emoji_mode="remove"),
    "norm_emoji_kept":     dict(normalize=True,  social=True,  emoji_mode="keep"),
    "norm_emoji_demojize": dict(normalize=True,  social=True,  emoji_mode="demojize"),
    "norm_emoji_category": dict(normalize=True,  social=True,  emoji_mode="category"),
    "norm_emoji_both":     dict(normalize=True,  social=True,  emoji_mode="both"),
    "norm_emoji_urdu":     dict(normalize=True,  social=True,  emoji_mode="urdu"),
    "norm_emoji_augment":  dict(normalize=True,  social=True,  emoji_mode="augment"),
}


if __name__ == "__main__":
    samples = [
        "یہ اسمبلی بغیر ڈیزل کے چل رہی ہے کیونکہ اِسکو گدھا '' چلا رہا ھے 🤣😂😂😂 ووٹ_کو_عزت_دو",
        "ہاہاہاہاہاہاہاہا ہنسی بھی نہیں آئی ؟؟؟؟ @kamran https://t.co/x 🤦‍♂️",
        "کدووووو 80روپے گز 😭😭😭 ISPR #StandWithSindhPolice",
    ]
    for s in samples:
        print(preprocess(s))
        print(preprocess(s, emoji_mode="urdu"))
        print(tokenize(normalize_urdu(s)))
