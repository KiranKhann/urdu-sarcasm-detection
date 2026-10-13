# -*- coding: utf-8 -*-
"""Re-draw EDA figures with an emoji-capable font (works on Windows; falls back to names)."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.emoji_semantics import LEXICON, describe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs"); FIG = os.path.join(OUT, "figures"); SPL = os.path.join(OUT, "splits")
EMOJI_FONT = None
for cand in [r"C:\Windows\Fonts\seguiemj.ttf", "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
             "/System/Library/Fonts/Apple Color Emoji.ttc"]:
    if os.path.exists(cand):
        try:
            EMOJI_FONT = font_manager.FontProperties(fname=cand); break
        except Exception:
            pass


def label(e):
    return e if EMOJI_FONT else LEXICON.get(e, {}).get("en", e)


df = pd.concat([pd.read_csv(os.path.join(SPL, f"{s}.csv")) for s in ("train", "val", "test")])
W = df["urdu_text"].str.split().str.len()

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
df["is_sarcastic"].value_counts().sort_index().plot.bar(ax=ax[0], color=["#4C72B0", "#DD8452"])
ax[0].set_title("Class distribution (after cleaning)"); ax[0].set_xlabel("")
ax[0].set_xticklabels(["non-sarcastic (0)", "sarcastic (1)"], rotation=0)
for p in ax[0].patches:
    ax[0].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom", fontsize=8)
for lab, c in [(0, "#4C72B0"), (1, "#DD8452")]:
    W[df["is_sarcastic"] == lab].plot.hist(bins=40, alpha=.6, ax=ax[1], color=c, label=f"class {lab}", density=True)
ax[1].set_title("Words per post (density)"); ax[1].legend(); ax[1].set_xlabel("words")
top = pd.Series([e.emoji for t in df["urdu_text"] for e in describe(t)]).value_counts().head(15)
ax[2].barh(range(len(top)), top.values, color="#55A868")
ax[2].set_yticks(range(len(top)))
ax[2].set_yticklabels([label(e) for e in top.index], fontproperties=EMOJI_FONT, fontsize=12)
ax[2].invert_yaxis(); ax[2].set_title("Most frequent emojis (all posts)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "eda.png"), dpi=150); plt.close()

# emoji sarcasm rate (training split only, n >= 40)
er = pd.read_csv(os.path.join(OUT, "emoji_sarcasm_rate.csv")).sort_values("sarcastic_rate")
base = pd.read_csv(os.path.join(SPL, "train.csv"))["is_sarcastic"].mean()
sel = pd.concat([er.head(12), er.tail(12)])
fig, ax = plt.subplots(figsize=(7, 6.5))
colors = ["#4C72B0" if r < base else "#DD8452" for r in sel["sarcastic_rate"]]
ax.barh(range(len(sel)), sel["sarcastic_rate"], color=colors)
ax.set_yticks(range(len(sel)))
ax.set_yticklabels([f"{label(e)}  (n={n})" for e, n in zip(sel["emoji"], sel["n"])], fontproperties=EMOJI_FONT, fontsize=11)
ax.axvline(base, color="k", ls="--", lw=1); ax.text(base + .01, len(sel) - .5, f"base rate {base:.2f}", fontsize=8)
ax.set_xlabel("P(sarcastic | emoji present)  - training split"); ax.set_title("Emojis most / least associated with sarcasm")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "emoji_sarcasm_rate.png"), dpi=150); plt.close()
print("figures written")
