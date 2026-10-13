# -*- coding: utf-8 -*-
"""
Hand-crafted, linguistically motivated features for Urdu sarcasm.

These complement the n-gram models with signals that bag-of-words cannot see:
  * emoji semantics (see emoji_semantics.py) - counts, polarity, position, intensity
  * word <-> emoji polarity CONTRAST  (positive words + negative emoji or vice-versa)
  * Urdu sarcasm markers (واہ, ماشاءاللہ, زبردست ... used ironically)
  * Urdu idioms / multi-word phrases that are typical of mockery (bonus: phrase features)
  * surface cues: question marks, ellipsis, repetition, quotes, laughter, code-mixing
"""
from __future__ import annotations

import re
from typing import List

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .emoji_semantics import EMOJI_FEATURE_NAMES, EmojiPrior, describe, emoji_features
from .preprocessing import normalize_urdu

# --------------------------------------------------------------------------- #
# small, hand-built Urdu lexicons                                             #
# --------------------------------------------------------------------------- #
POSITIVE_WORDS = set("""
واہ زبردست شاندار بہترین عظیم کمال خوبصورت شکریہ مبارک ماشاءاللہ ماشاءالله سبحان الحمدللہ خوب
شاباش ویلڈن بہادر قابل فخر محبت پیار اچھا اچھی اچھے بہت خوش خوشی عزت کامیاب کامیابی جیت
سلام سلیوٹ دعا رحمت برکت سکون جنت مسکراہٹ ہیرو لیڈر قائد عظمت بے مثال لاجواب پسند
""".split())

NEGATIVE_WORDS = set("""
بکواس ذلیل جاہل بیوقوف بے وقوف گدھا گدھے کتا کتے چور ڈاکو لٹیرے جھوٹ جھوٹا جھوٹے منافق بے غیرت
بیغیرت شرم بے شرم نالائق نکما نکمے بدتمیز بدمعاش غدار غلام غلامی کمینہ کمینے کھوتا کھوتی پاگل
نشئی بھکاری چاپلوس چاپلوسی لفافی لفافہ ڈرامہ ڈرامے ٹوپی ڈرامہ بزدل مہنگائی تباہ برباد بدبو
مذمت افسوس شرمناک ظلم ظالم موٹی گندی گندا جعلی فراڈ
""".split())

NEGATION_WORDS = set("نہیں نہ مت نا نئیں نی ناں کبھی ہرگز بغیر بنا".split())

# words/phrases that are sincere praise literally but are the stock vehicles of
# Urdu irony ("واہ خان صاحب", "ماشاءاللہ کیا بات ہے", "تبدیلی آ گئی")
SARCASM_MARKERS = [
    "واہ", "واہ جی", "واہ بھئی", "واہ واہ", "ماشاءاللہ", "سبحان اللہ", "کیا بات ہے", "کیا کہنے",
    "بہت خوب", "زبردست", "کمال", "شاباش", "ویلڈن", "مبارک ہو", "عظیم", "جناب", "صاحب",
    "تبدیلی", "نیا پاکستان", "ریاست مدینہ", "یوتھیے", "پٹواری", "لفافی", "سلیکٹڈ", "کپتان",
    "خان صاحب", "نوٹس", "ٹوپی ڈرامہ", "لے بھئی", "لو جی", "چلو جی", "اور کیا", "یہی تو",
    "ہاں جی", "بالکل", "ظاہر ہے", "بے شک", "آخر", "بس", "تو پھر", "کیا کریں", "مجال ہے",
]
SARCASM_MARKER_RE = re.compile("|".join(r"(?<![؀-ۿ])" + re.escape(m) + r"(?![؀-ۿ])"
                                        for m in sorted(SARCASM_MARKERS, key=len, reverse=True)))

# Urdu / Punjabi idioms and multi-word expressions (bonus: phrase features).
# Idioms are a classic irony vehicle: the literal meaning is absurd, the figurative
# meaning is mockery.  Each entry is a regex over normalised text.
IDIOMS = {
    "اپنے پیر پر کلہاڑی": "shoot oneself in the foot",
    "چراغ تلے اندھیرا": "darkness under the lamp (hypocrisy)",
    "دودھ کا راکھا بلا": "cat guarding the milk (fox guarding henhouse)",
    "اونٹ کے منہ میں زیرہ": "a drop in the ocean",
    "آ بیل مجھے مار": "inviting trouble",
    "ناچ نہ جانے آنگن ٹیڑھا": "bad workman blames his tools",
    "اندھا بانٹے ریوڑیاں": "nepotism",
    "ٹوپی ڈرامہ": "staged drama",
    "کر لو گل": "so what now (taunt)",
    "ہور کی": "what else (Punjabi taunt)",
    "لگے رہو": "keep it up (ironic)",
    "منہ کالا": "disgrace",
    "گھر کی مرغی دال برابر": "familiarity breeds contempt",
    "بندر کیا جانے ادرک کا سواد": "pearls before swine",
    "جیتھے دی کھوتی": "the donkey returns where it stood (Punjabi)",
    "کھسیانی بلی کھمبا نوچے": "sour grapes",
    "تھوک کے چاٹ": "eating ones words",
    "چور مچائے شور": "the thief shouts loudest",
    "سانپ نکل گیا لکیر پیٹ": "closing the barn door after the horse",
    "اندھیر نگری": "lawless city",
    "دال میں کچھ کالا": "something fishy",
    "ہاتھی کے دانت": "elephant teeth (hypocrisy)",
    "آسمان سے گرا کھجور میں اٹکا": "out of the frying pan into the fire",
    "نو سو چوہے کھا کر بلی حج": "late piety after a life of sin",
    "سر پر چڑھانا": "spoil someone",
    "منہ توڑ جواب": "crushing reply",
    "الٹا چور کوتوال کو ڈانٹے": "the thief scolds the policeman",
    "رہنے دے": "let it be (dismissive)",
    "لے بھئی": "well well (taunt)",
    "مجال ہے": "as if (ironic)",
    "خدا کی پناہ": "god forbid",
    "اللہ معاف کرے": "god forgive (ironic)",
    "مطلب": "meaning (that is to say)",
}
IDIOM_RE = re.compile("|".join(re.escape(k) for k in sorted(IDIOMS, key=len, reverse=True)))

_LATIN = re.compile(r"[A-Za-z]{2,}")
_URDU_WORD = re.compile(r"[؀-ۿ]+")
_LAUGH = re.compile(r"(?:ہا|ھا|ہہ|ha|he){3,}|lol|lmao", re.I)
_ELONG = re.compile(r"(.)\1{2,}")
_QUOTE = re.compile("``|''|\"|“|”")

FEATURE_NAMES = [
    "n_chars", "n_tokens", "n_urdu_words", "n_latin_words", "latin_ratio", "n_digits",
    "n_question", "n_exclaim", "n_ellipsis", "n_repeat_punct", "n_elongation", "has_laughter",
    "has_quote", "has_hashtag", "has_mention", "has_url", "n_pos_words", "n_neg_words",
    "n_negation", "text_polarity", "n_sarcasm_markers", "n_idioms",
    "contrast_posword_negemoji", "contrast_negword_posemoji", "contrast_posword_cue_emoji",
    "contrast_any", "marker_with_laugh_emoji",
] + EMOJI_FEATURE_NAMES


def handcrafted(text: str) -> List[float]:
    raw = "" if text is None else str(text)
    t = normalize_urdu(raw)
    toks = t.split()
    urdu_words = _URDU_WORD.findall(t)
    latin = _LATIN.findall(t)
    n_tok = max(len(toks), 1)
    pos = sum(w in POSITIVE_WORDS for w in urdu_words)
    neg = sum(w in NEGATIVE_WORDS for w in urdu_words)
    negation = sum(w in NEGATION_WORDS for w in urdu_words)
    markers = len(SARCASM_MARKER_RE.findall(t))
    idioms = len(IDIOM_RE.findall(t))
    emo = describe(raw)
    emo_pos = sum(e.polarity > 0 for e in emo)
    emo_neg = sum(e.polarity < 0 for e in emo)
    emo_cue = sum(e.cue for e in emo)
    emo_laugh = sum(e.category == "LAUGH" for e in emo)
    c1 = float(pos > 0 and emo_neg > 0)
    c2 = float(neg > 0 and emo_pos > 0)
    c3 = float((pos > 0 or markers > 0) and emo_cue > 0)
    feats = [
        len(raw), len(toks), len(urdu_words), len(latin), len(latin) / n_tok,
        sum(ch.isdigit() for ch in t),
        t.count("?") + t.count("؟"), t.count("!"),
        len(re.findall(r"\.{2,}|۔{2,}|…", t)), len(re.findall(r"([!?؟])\1+", t)),
        len(_ELONG.findall(t)), float(bool(_LAUGH.search(t))),
        float(bool(_QUOTE.search(t))), float("#" in t or bool(re.search(r"\w_\w", t))),
        float("@" in t), float(bool(re.search(r"http|www\.", t))),
        pos, neg, negation, (pos - neg) / n_tok, markers, idioms,
        c1, c2, c3, float(c1 or c2 or c3), float(markers > 0 and emo_laugh > 0),
    ] + emoji_features(raw)
    return [float(x) for x in feats]


class HandcraftedFeatures(BaseEstimator, TransformerMixin):
    """sklearn transformer: raw texts -> dense matrix of linguistic + emoji features.

    Also fits an EmojiPrior on the training texts (data-driven emoji log-odds),
    so the only state learnt is from the TRAIN split passed to fit()."""

    def __init__(self, use_prior: bool = True, log_scale: bool = True):
        self.use_prior = use_prior
        self.log_scale = log_scale

    def fit(self, X, y=None):
        self.prior_ = EmojiPrior().fit(X, y) if (self.use_prior and y is not None) else None
        return self

    def transform(self, X):
        M = np.array([handcrafted(t) for t in X], dtype=np.float32)
        if self.log_scale:               # compress heavy-tailed counts
            M = np.sign(M) * np.log1p(np.abs(M))
        if self.prior_ is not None:
            P = np.array(self.prior_.transform(X), dtype=np.float32)
            M = np.hstack([M, P])
        return M

    def get_feature_names_out(self, input_features=None):
        names = list(FEATURE_NAMES)
        if self.prior_ is not None:
            names += EmojiPrior.FEATURE_NAMES
        return np.array(names)


def explain_features(text: str) -> dict:
    """Return the non-zero handcrafted features of a text (for error analysis)."""
    vals = handcrafted(text)
    return {n: v for n, v in zip(FEATURE_NAMES, vals) if v}


if __name__ == "__main__":
    s = "واہ خان صاحب کیا بات ہے تبدیلی آ گئی 😂😂 ووٹ_کو_عزت_دو"
    print(explain_features(s))
