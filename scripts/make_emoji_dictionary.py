# -*- coding: utf-8 -*-
"""Build the emoji-meaning dictionary for the emojis that occur in the dataset
(outputs/emoji_meaning_dictionary.csv): meaning in English and Urdu, polarity,
pragmatic category, irony-cue flag, corpus frequency and P(sarcastic | emoji) on the training split."""
import os
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.emoji_semantics import LEXICON, describe, meaning, _lookup

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs"); SPL = os.path.join(OUT, "splits")
train = pd.read_csv(os.path.join(SPL, "train.csv"))
allp = pd.concat([pd.read_csv(os.path.join(SPL, f"{s}.csv")) for s in ("train", "val", "test")])

freq = Counter(e.emoji for t in allp["urdu_text"] for e in describe(t))
pos, tot = Counter(), Counter()
for t, y in zip(train["urdu_text"], train["is_sarcastic"]):
    for e in {x.emoji for x in describe(t)}:
        tot[e] += 1; pos[e] += int(y)
rows = []
for e, n in freq.most_common():
    info = _lookup(e)
    rows.append({"emoji": e, "meaning_en": info["en"], "meaning_ur": info["ur"], "polarity": info["pol"],
                 "category": info["cat"], "irony_cue": info["cue"], "in_curated_lexicon": e in LEXICON,
                 "count_all_posts": n, "train_posts": tot[e],
                 "P_sarcastic_given_emoji_train": round(pos[e] / tot[e], 3) if tot[e] else None})
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "emoji_meaning_dictionary.csv"), index=False, encoding="utf-8")
print(len(df), "distinct emojis;", df["in_curated_lexicon"].sum(), "with curated Urdu meaning;",
      "coverage of emoji occurrences:", round(df[df.in_curated_lexicon]["count_all_posts"].sum() / df["count_all_posts"].sum(), 3))
print(df.head(40).to_string())
