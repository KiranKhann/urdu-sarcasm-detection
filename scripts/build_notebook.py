# -*- coding: utf-8 -*-
"""Assemble the self-contained assignment notebook (Colab-ready) from the src/ modules
and the results in outputs/.  Run:  python scripts/build_notebook.py"""
import json
import os

import nbformat as nbf
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src"); OUT = os.path.join(ROOT, "outputs")
NB_PATH = os.path.join(ROOT, "notebooks", "Urdu_Sarcasm_Detection.ipynb")

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))


def read(name):
    return open(os.path.join(SRC, name), encoding="utf-8").read()


def table(path, cols=None, fmt=3):
    if not os.path.exists(path):
        return "_(not available)_"
    df = pd.read_csv(path)
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    return df.round(fmt).to_markdown(index=False)


# =========================================================================== #
md("""
# Urdu Sarcasm Detection — DS5007 NLP, Programming Assignment 1

**Task.** Binary classification of Urdu social-media posts: `is_sarcastic` = 1 (sarcastic) / 0 (non-sarcastic).

**What is special about this solution**

* **Emojis are treated as meaning, not noise.** 80 % of the posts contain emojis. A curated lexicon
  (`src/emoji_semantics.py`) gives every emoji an English name, an **Urdu gloss**, a sentiment polarity,
  a pragmatic category (LAUGH, TEASE, SMIRK, THINK, LOVE, SAD, ANGER, PRAYER, PRAISE, FLOWER, SHY, GESTURE, INSULT)
  and an *irony-cue* flag. The lexicon feeds (i) token rewriting for n-gram models, (ii) an Urdu-gloss
  input mode for the transformer, (iii) dense features such as **word–emoji polarity contrast**
  (praise words + 😂/🙄) and (iv) a data-driven emoji sarcasm prior learnt on the training split only.
* **Urdu-aware preprocessing** that is conservative and fully ablated: Arabic/Urdu code-point unification,
  diacritic removal, keyboard spelling fixes (ھے→ہے), hashtag reconstruction (ووٹ_کو_عزت_دو), laughter
  and elongation normalisation — while *keeping* punctuation, repetition, Roman/English tokens and Punjabi.
* **Hand-crafted Urdu phrase features** (bonus): ironic-praise markers (واہ, ماشاءاللہ, کیا بات ہے …) and
  ~35 idioms (اپنے پیر پر کلہاڑی, چراغ تلے اندھیرا, ٹوپی ڈرامہ …).
* Three model families: **TF-IDF (LR / SVM / NB)**, **TextCNN & BiLSTM+attention (PyTorch)**, **XLM-RoBERTa fine-tuning**,
  compared with Macro-F1, plus a **5-fold cross-validated ablation study** and a linguistic **error analysis**.

> The notebook is self-contained: section 1 writes the `src/` package to disk, so it runs on Google Colab
> (Runtime → GPU recommended for sections 6–7) or locally (`pip install -r requirements.txt`).
""")

# --------------------------------------------------------------------------- 0. setup
md("## 0. Setup")
code("""
import sys, subprocess, importlib
def _ensure(pkgs):
    missing = [p for p in pkgs if importlib.util.find_spec(p.split('==')[0].replace('-', '_')) is None]
    if missing:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *missing])
_ensure(['emoji', 'scikit-learn', 'pandas', 'matplotlib', 'joblib', 'tabulate'])
_ensure(['torch', 'transformers', 'sentencepiece'])   # for sections 6-7

import os, json, re, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
pd.set_option('display.max_colwidth', 160); pd.set_option('display.width', 220)
os.makedirs('src', exist_ok=True); os.makedirs('outputs/figures', exist_ok=True)
RUN_TRANSFORMER = True          # set False to skip section 7 (needs a GPU; ~10 min on a T4)
SEED = 42
""")

md("""
### 0.1 Load the dataset
On Colab you will be prompted to upload `urdu_sarcastic_dataset.csv` (columns `urdu_text`, `is_sarcastic`).
Locally the file is read from `data/`.
""")
code("""
DATA_PATH = None
for cand in ['data/urdu_sarcastic_dataset.csv', '../data/urdu_sarcastic_dataset.csv', 'urdu_sarcastic_dataset.csv']:
    if os.path.exists(cand):
        DATA_PATH = cand; break
if DATA_PATH is None:
    try:
        from google.colab import files
        print('Please upload urdu_sarcastic_dataset.csv'); up = files.upload()
        DATA_PATH = list(up.keys())[0]
    except ImportError:
        raise FileNotFoundError('put urdu_sarcastic_dataset.csv in data/')
raw = pd.read_csv(DATA_PATH, encoding='utf-8')
print(raw.shape); raw.head()
""")

# --------------------------------------------------------------------------- 1. source package
md("""
## 1. The `src` package (written to disk so the notebook is self-contained)

Each cell below is one module. They are shown in full because the design decisions in them are part of the
assignment: how emojis are interpreted, how Urdu text is normalised, and which hand-crafted features are extracted.
""")
code("%%writefile src/__init__.py\n# Urdu sarcasm detection package")
for fname, title in [("emoji_semantics.py", "1.1 Emoji semantics — lexicon, token rewriting, features, data-driven prior"),
                     ("preprocessing.py", "1.2 Urdu-aware normalisation and tokenisation"),
                     ("features.py", "1.3 Hand-crafted linguistic + emoji features (incl. idioms / phrases)"),
                     ("data.py", "1.4 Cleaning and leakage-free stratified splitting"),
                     ("evaluate.py", "1.5 Metrics, confusion matrices and error-analysis helpers"),
                     ("baselines.py", "1.6 TF-IDF baselines and ablation runner"),
                     ("neural.py", "1.7 PyTorch TextCNN and BiLSTM + attention"),
                     ("transformer.py", "1.8 Transformer fine-tuning loop"),
                     ("predict.py", "1.9 Predict + explain: the label together with the meaning of every emoji")]:
    md(f"### {title}")
    code(f"%%writefile src/{fname}\n" + read(fname))

code("""
import importlib, sys
sys.path.insert(0, '.')
for m in list(sys.modules):
    if m.startswith('src'): del sys.modules[m]
from src.emoji_semantics import describe, explain, meaning, emoji_to_tokens, emoji_features, EmojiPrior, LEXICON, EMOJI_FEATURE_NAMES
from src.predict import SarcasmExplainer, format_explanation
from src.preprocessing import normalize_urdu, normalize_social, tokenize, preprocess, PREPROCESS_VARIANTS
from src.features import handcrafted, explain_features, HandcraftedFeatures, FEATURE_NAMES, IDIOMS, SARCASM_MARKERS
from src.data import clean, split, TEXT, LABEL
from src.evaluate import evaluate, results_table, plot_confusion, error_table, phenomena, phenomena_summary
from src.baselines import make_pipeline, fit_eval, predict_proba, run_ablations, top_features, ABLATIONS
print('src package loaded;', len(LEXICON), 'emojis in lexicon')
""")

# --------------------------------------------------------------------------- 2. EDA
md("""
## 2. Exploratory data analysis

### 2.1 Missing values, duplicates, conflicting labels
The sheet export carries empty `Unnamed:` columns and a few rows without text. More importantly the file contains
many **exact duplicate posts** and **posts that appear with both labels**. Duplicates would leak between train and test;
conflicting labels are unlearnable, so both are removed *before* splitting.
""")
code("""
df, report = clean(raw[[TEXT, LABEL]])
print(json.dumps(report, indent=1, ensure_ascii=False))
train, val, test = split(df, seed=SEED)
print({k: len(v) for k, v in dict(train=train, val=val, test=test).items()},
      '| sarcastic share:', round(train[LABEL].mean(), 3), round(val[LABEL].mean(), 3), round(test[LABEL].mean(), 3))
# a few of the conflicting-label posts (annotation noise)
dup = raw[[TEXT, LABEL]].dropna(); g = dup.groupby(TEXT)[LABEL].nunique()
dup[dup[TEXT].isin(g[g > 1].index)].sort_values(TEXT).head(6)
""")
md("""
### 2.2 Class distribution, text length, emoji usage
After de-duplication the classes are no longer perfectly balanced (59 % / 41 %): the sarcastic class had far more
duplicated rows. Sarcastic posts are also **longer** on average (median 20 vs 13 words) — mockery needs a set-up —
and, surprisingly, use **fewer emojis** (67 % vs 88 % of posts), because the non-sarcastic class is dominated by
poetry/greetings decorated with ❤ 🌹 💔.
""")
code("""
L = df[TEXT].str.len(); W = df[TEXT].str.split().str.len()
print(pd.DataFrame({'chars': L.describe(), 'words': W.describe()}).round(1).T)
print(W.groupby(df[LABEL]).median().rename('median words by class'))
has_emoji = df[TEXT].apply(lambda t: len(describe(t)) > 0)
print('share with emoji:', round(has_emoji.mean(), 3), '| by class:', has_emoji.groupby(df[LABEL]).mean().round(3).to_dict())
print('share with Latin-script tokens:', round(df[TEXT].str.contains(r'[A-Za-z]{2,}').mean(), 3),
      '| hashtags:', round(df[TEXT].str.contains(r'#|\\w_\\w').mean(), 3),
      '| mentions:', round(df[TEXT].str.contains('@').mean(), 3),
      '| URLs:', round(df[TEXT].str.contains(r'http|www\\.').mean(), 3),
      '| quoted speech (`` ... \\'\\'):', round(df[TEXT].str.contains("``|''").mean(), 3))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
df[LABEL].value_counts().sort_index().plot.bar(ax=ax[0], color=['#4C72B0', '#DD8452']); ax[0].set_title('class distribution'); ax[0].set_xticklabels(['non-sarcastic', 'sarcastic'], rotation=0)
for lab, c in [(0, '#4C72B0'), (1, '#DD8452')]:
    W[df[LABEL] == lab].plot.hist(bins=40, alpha=.6, ax=ax[1], color=c, label=f'class {lab}', density=True)
ax[1].set_title('words per post'); ax[1].legend(); plt.tight_layout(); plt.show()
""")
md("""
### 2.3 Representative examples and what the emojis mean
`explain()` prints, for every emoji, its meaning (English / Urdu), polarity, pragmatic category and whether it is an
irony cue. Note the typical sarcastic pattern: a literal statement of praise or a political claim followed by 😂 / 😜 / 🙄.
""")
code("""
for lab in (1, 0):
    print(f'\\n===== {"SARCASTIC" if lab else "NON-SARCASTIC"} examples =====')
    for t in train[train[LABEL] == lab][TEXT].sample(5, random_state=3):
        print('•', t[:140]); print('   emojis:', explain(t))
""")
md("""
### 2.4 Which emojis signal sarcasm in this corpus?
P(sarcastic | emoji present) on the **training split only**. Hearts, flowers and the broken heart (shayari/greeting
register) almost never co-occur with sarcasm, while 😅 🤔 😉 😁 😆 are well above the base rate. 😂, the most frequent emoji,
is only mildly informative on its own — it marks *humour*, and the model must look at the words to decide whether the
laughter is *at* someone.
""")
code("""
prior = EmojiPrior().fit(train[TEXT], train[LABEL])
cnt = {}
for t, y in zip(train[TEXT], train[LABEL]):
    for e in {x.emoji for x in describe(t)}: cnt.setdefault(e, [0, 0])[int(y)] += 1
er = pd.DataFrame([{'emoji': e, 'meaning': LEXICON.get(e, {}).get('en', '?'), 'n': a + b, 'P(sarcastic|emoji)': b / (a + b)}
                   for e, (a, b) in cnt.items() if a + b >= 40]).sort_values('P(sarcastic|emoji)')
print('base rate:', round(train[LABEL].mean(), 3))
display(pd.concat([er.head(10), er.tail(10)]).round(3))
""")
md("""
### 2.4b Emoji meaning dictionary for this corpus
Every emoji that occurs in the data, with its meaning in **English and Urdu**, polarity, pragmatic category, irony-cue flag,
frequency and P(sarcastic | emoji) on the training split. The curated lexicon covers 96 % of all emoji occurrences; the rest fall
back to the Unicode name with neutral polarity. (Saved to `outputs/emoji_meaning_dictionary.csv`.)
""")
code("""
from collections import Counter
from src.emoji_semantics import _lookup
freq = Counter(e.emoji for t in df[TEXT] for e in describe(t))
emo_dict = pd.DataFrame([{'emoji': e, 'meaning_en': _lookup(e)['en'], 'meaning_ur': _lookup(e)['ur'], 'polarity': _lookup(e)['pol'],
                          'category': _lookup(e)['cat'], 'irony_cue': _lookup(e)['cue'], 'count': n,
                          'P(sarcastic|emoji)': round(cnt[e][1] / sum(cnt[e]), 3) if e in cnt else None}
                         for e, n in freq.most_common()])
emo_dict.to_csv('outputs/emoji_meaning_dictionary.csv', index=False, encoding='utf-8')
print(len(emo_dict), 'distinct emojis; curated coverage of occurrences:',
      round(emo_dict[emo_dict.emoji.isin(LEXICON)]['count'].sum() / emo_dict['count'].sum(), 3))
display(emo_dict.head(30))
""")
md("""
### 2.5 Data-quality issues found
1. **49 rows without text**, 4 095 **exact duplicates** (mostly sarcastic), **646 texts labelled both 0 and 1** — all dropped.
2. **Collection artefacts**: some posts still contain the hashtags used to harvest them (`#طنز`, `#مزاح`, `#sarcasm`). A
   linear model learns these as the strongest sarcasm features (see 4.4); we test removing them in the ablation.
3. **Label noise / ambiguity**: many clearly mocking political posts are labelled 0 and some plain shayari with 💔 are labelled 1
   (see the error analysis). This caps the achievable accuracy for every model.
4. **Script and spelling variation**: Arabic vs Urdu code points (ك/ک, ي/ی), ھ for ہ (ھے), Sindhi/Pashto forms (ناٸیٹ), diacritics,
   elongation (کدووووو), and Punjabi written in Urdu script (~8 % of posts).
5. **Code-mixing**: 15 % of posts contain Latin-script tokens (ISPR, PM, hashtags); 7 % have hashtags whose `#` was lost in scraping
   and survive as `words_joined_by_underscores`.
""")

# --------------------------------------------------------------------------- 3. preprocessing
md("""
## 3. Urdu-aware preprocessing

Every step is motivated by an observation in §2 and is switchable so it can be ablated:

| step | why |
|---|---|
| Unicode NFC + code-point unification (ك→ک, ي→ی, ة→ہ, أ→ا, Sindhi ٸ→ئ), digit unification | the same word is typed with Arabic, Persian or Urdu keyboards; otherwise n-grams fragment |
| diacritic / tashkeel removal | used decoratively in religious posts and shayari; carries no lexical information |
| keyboard spelling fixes (ھے→ہے, ھیں→ہیں, کوئ→کوئی …) | very frequent in the data; ھ is a different letter (تھا) so only whole-word fixes are applied |
| URL → `URL`, @user → `@USER` | identity of the account is not a sarcasm cue; presence of a mention might be |
| hashtag reconstruction `ووٹ_کو_عزت_دو` → `HASHTAG ووٹ کو عزت دو` | the words of the slogan are informative, the underscores are not |
| laughter (ہاہاہاہا, hahaha, lol) → `LAUGH`, elongation (کدووووو → کدوو) | normalise open-class repetition, keep the fact that it happened |
| Urdu punctuation → ASCII, repeated `؟؟؟` → `???` kept as a token | rhetorical questions and ellipsis are sarcasm cues, so punctuation is **kept** |
| emojis → **meaning tokens** (never removed by default) | see §1.1; raw symbols are fine for char n-grams but word models need a token |
| emoji **augment** mode: `😂` → `😂 ہنسی کے آنسو` | the model sees the symbol *and* its meaning in Urdu; default input for the transformer |

Emojis are protected before normalisation because the zero-width joiner we strip from Urdu text is part of emoji sequences (🤦‍♂️).
Tokenisation is regex based (Urdu word runs, Latin words, numbers, punctuation runs, single emojis) — Urdu has no reliable
off-the-shelf tokenizer for social-media text and whitespace is unreliable (اسکو vs اس کو).
""")
code("""
samples = ['یہ اسمبلی بغیر ڈیزل کے چل رہی ہے کیونکہ اِسکو گدھا \\'\\' چلا رہا ھے 🤣😂😂😂 ووٹ_کو_عزت_دو',
           'ہاہاہاہاہاہاہاہا ہنسی بھی نہیں آئی ؟؟؟؟ @kamran https://t.co/x 🤦‍♂️',
           'واہ خان صاحب کیا بات ہے تبدیلی آ گئی 😍😂 ISPR']
for s in samples:
    print('RAW      :', s)
    for mode in ['remove', 'keep', 'demojize', 'category', 'urdu', 'augment', 'both']:
        print(f'{mode:9s}:', preprocess(s, emoji_mode=mode))
    print('features :', explain_features(s)); print()
""")

# --------------------------------------------------------------------------- 4. classical
md("""
## 4. Classical baseline: TF-IDF + linear models

Feature groups: **word uni/bi-grams**, **character 2–5-grams within word boundaries** (robust to the spelling variation
above and to Urdu's rich suffix morphology) and the **dense hand-crafted block** (§1.3, standardised). Logistic Regression
(liblinear, class-balanced), a calibrated Linear SVM, and Complement Naive Bayes are compared on the validation set;
Macro-F1 is the selection metric. Hyper-parameters are chosen on validation only; the test set is touched once at the end.
""")
code("""
X_tr, y_tr = train[TEXT].tolist(), train[LABEL].values
X_va, y_va = val[TEXT].tolist(), val[LABEL].values
X_te, y_te = test[TEXT].tolist(), test[LABEL].values
val_results, pipes = [], {}
for clf, Cs in [('lr', [1, 2, 4]), ('svm', [0.1, 0.3, 1]), ('nb', [0.1, 0.3])]:
    for C in Cs:
        name = f'{clf.upper()} C={C}'
        pipes[name] = make_pipeline('word+char+hand', clf=clf, C=C, emoji_mode='both')
        r = fit_eval(pipes[name], X_tr, y_tr, X_va, y_va, name); r.update(clf=clf, C=C); val_results.append(r)
        print(f'{name:12s} val macro-F1 {r["macro_f1"]:.4f}  acc {r["accuracy"]:.4f}')
val_df = pd.DataFrame(val_results)
""")
md("### 4.1 Test-set results of the classical models")
code("""
classical = []
for clf in ['lr', 'svm', 'nb']:
    best = val_df[val_df.clf == clf].sort_values('macro_f1').iloc[-1]; pipe = pipes[best['model']]
    name = f'TF-IDF {clf.upper()} (word+char+hand)' if clf != 'nb' else 'TF-IDF ComplementNB (word+char)'
    pred = pipe.predict(X_te); r = evaluate(y_te, pred, name); classical.append(r)
    if clf == 'lr': lr_pipe, lr_pred, lr_prob = pipe, pred, predict_proba(pipe, X_te)
display(results_table(classical))
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
for ax, r in zip(axes, classical): plot_confusion(r, ax=ax)
plt.tight_layout(); plt.show()
""")
md("""
### 4.2 Ablation study (validation split)
Three questions: **(A)** how should emojis be represented? **(B)** which feature family matters? **(C)** does normalisation help?
""")
code("""
abl = run_ablations(X_tr, y_tr, X_va, y_va, clf='lr', C=2.0)
abl_df = pd.DataFrame(abl)[['model', 'accuracy', 'macro_f1', 'f1_0', 'f1_1', 'n_features']]
display(abl_df.round(4))
""")
md("""
### 4.3 Cross-validated ablation (5 folds on train+val)
Single-split differences of a few tenths of a point are within noise (validation n = 2 186 ⇒ ±~1 pt). The table below
was produced by `scripts/run_cv_ablation.py` (mean ± std over 5 stratified folds, LR, C = 2). Re-run it here if desired (≈ 15 min):
`!python scripts/run_cv_ablation.py`.
""" + "\n\n" + table(os.path.join(OUT, "ablations_cv.csv"), ["config", "macro_f1_mean", "macro_f1_std", "acc_mean"], 4) + """

**Reading the ablation.**
* *Emojis*: removing emojis costs little for the n-gram model on this corpus — once words such as مزاح / طنز or a political
  name are present, the emoji is redundant; where emojis help is on **short posts** whose words are neutral (see error analysis).
  Representing the emoji by its *meaning* (demojize / Urdu gloss) is as good as or slightly better than the raw symbol, which
  is only visible to character n-grams.
* *Features*: word and char n-grams are complementary; the hand-crafted block alone reaches ≈ 0.75 macro-F1 with only ~70
  features, and adds a small, consistent gain on top of n-grams.
* *Normalisation*: Urdu character normalisation helps; the more aggressive social-media normalisation (laughter / punctuation
  spacing) is neutral-to-slightly-negative, so the final models use it only for the hashtag/URL/mention steps.
* *Artefact hashtags*: removing `#طنز/#مزاح/#sarcasm` lowers the score, confirming they leak the label — reported scores
  with these tokens present are optimistic relative to true in-the-wild performance.
""")
md("### 4.4 What the linear model learnt")
code("""
tf = top_features(lr_pipe, 25)
print('MOST SARCASTIC  :', [f'{n} ({w})' for n, w in tf['sarcastic']])
print('\\nLEAST SARCASTIC :', [f'{n} ({w})' for n, w in tf['non_sarcastic']])
hand = [(n, w) for n, w in tf['sarcastic'] + tf['non_sarcastic'] if n.startswith('hand:')]
print('\\nhand-crafted features among the top weights:', hand)
""")

# --------------------------------------------------------------------------- 5/6. neural
md("""
## 5. Neural models (PyTorch)

* **TextCNN** — embeddings (200-d, learnt from scratch) → parallel 1-D convolutions of width 2/3/4/5 → max-over-time pooling.
  Acts as a learnt n-gram detector and sees emoji *sequences* (😂😂🙄) because every emoji is one token.
* **BiLSTM + attention** — bidirectional LSTM (128 hidden per direction) with additive attention pooling + max pooling;
  can relate a set-up at the beginning of the post to the punch-line emoji at the end.
* **Hybrid variants** concatenate the standardised hand-crafted / emoji feature vector (§1.3) to the pooled text representation.

Training: Adam, class-weighted cross-entropy, dropout 0.5, gradient clipping, early stopping on validation Macro-F1,
decision threshold tuned on validation. The emoji ablation is repeated for the neural models (tokens kept vs removed).
""")
code("""
import torch
from src.neural import Vocab, TextCNN, BiLSTMAttn, train_model, predict, best_threshold, set_seed, get_device
device = get_device(); print('device:', device)
MAX_LEN = 64
neural, hist_all = [], {}
for emoji_mode in ['keep', 'remove']:
    vocab = Vocab(min_freq=2, emoji_mode=emoji_mode).fit(train[TEXT])
    enc = lambda d: np.array([vocab.encode(t, MAX_LEN) for t in d[TEXT]], dtype=np.int64)
    Xn_tr, Xn_va, Xn_te = enc(train), enc(val), enc(test)
    for arch in ['cnn', 'bilstm']:
        for hybrid in ([False, True] if emoji_mode == 'keep' else [False]):
            set_seed(SEED)
            if hybrid:
                hf = HandcraftedFeatures().fit(train[TEXT], y_tr); D_tr = hf.transform(train[TEXT])
                mu, sd = D_tr.mean(0), D_tr.std(0) + 1e-6
                D_tr, D_va, D_te = [((hf.transform(d[TEXT]) - mu) / sd).astype(np.float32) for d in (train, val, test)]
                nd = D_tr.shape[1]
            else:
                D_tr = D_va = D_te = None; nd = 0
            model = TextCNN(len(vocab), n_dense=nd) if arch == 'cnn' else BiLSTMAttn(len(vocab), n_dense=nd)
            name = ('TextCNN' if arch == 'cnn' else 'BiLSTM+Attn') + (' + hand-crafted' if hybrid else '') + f' (emoji {emoji_mode})'
            model, hist = train_model(model, Xn_tr, y_tr, Xn_va, y_va, D_tr, D_va, epochs=20, patience=4, name=name, device=device, verbose=False)
            hist_all[name] = hist
            thr, f_va = best_threshold(y_va, predict(model, Xn_va, D_va, device=device))
            p_te = predict(model, Xn_te, D_te, device=device); pred = (p_te >= thr).astype(int)
            r = evaluate(y_te, pred, name); r.update(threshold=thr, val_macro_f1=f_va, epochs=len(hist)); neural.append(r)
            print(f'{name:45s} epochs {len(hist):2d}  val F1 {f_va:.4f}  test macro-F1 {r["macro_f1"]:.4f}  acc {r["accuracy"]:.4f}')
            if arch == 'bilstm' and hybrid: bilstm_pred, bilstm_prob = pred, p_te
            if arch == 'cnn' and hybrid: cnn_val_prob, cnn_test_prob = predict(model, Xn_va, D_va, device=device), p_te
display(results_table(neural))
fig, ax = plt.subplots(figsize=(6, 3.2))
for n, h in hist_all.items(): ax.plot([e['val_macro_f1'] for e in h], label=n)
ax.set_xlabel('epoch'); ax.set_ylabel('val macro-F1'); ax.legend(fontsize=7); plt.tight_layout(); plt.show()
""")

# --------------------------------------------------------------------------- 7. transformer
md("""
## 6. Transformer: fine-tuning XLM-RoBERTa-base

`xlm-roberta-base` (Conneau et al., 2020) was pre-trained on 2.5 TB of CommonCrawl text in 100 languages including Urdu,
so it brings sub-word knowledge of Urdu morphology, Roman-Urdu/English code-mixing and (for frequent ones) emojis.
We fine-tune the full model with a classification head (AdamW, lr 2e-5, linear warm-up, class-weighted loss, fp16, 3 epochs,
early stopping on validation Macro-F1, threshold tuned on validation).

The 250k × 768 word-embedding matrix (192M of 278M parameters) is frozen: this fits full fine-tuning of the 12 encoder layers into a
6 GB GPU and regularises training on 10k examples.

Emoji ablation for the transformer: **augment** (emoji + its Urdu meaning: 😂 → 😂 ہنسی کے آنسو — the model sees the symbol as
used *and* what it means; our default), **keep** (raw emoji; XLM-R's SentencePiece vocabulary has single pieces for 😂 ❤ 😭 … and
byte-fallback for rare ones) and **Urdu gloss only** (😂 → ہنسی کے آنسو).
""")
code("""
transformer = []
if RUN_TRANSFORMER:
    from src.transformer import fine_tune, predict_texts
    print('GPU available:', torch.cuda.is_available())
    for emoji_mode in ['augment', 'keep', 'urdu']:
        model, tok, hist, p_va = fine_tune('xlm-roberta-base', train[TEXT], y_tr, val[TEXT], y_va, emoji_mode=emoji_mode,
                                           epochs=3, max_len=96, bs=16, verbose=False)
        thr, f_va = best_threshold(y_va, p_va)
        p_te = predict_texts(model, tok, test[TEXT], emoji_mode=emoji_mode, max_len=96); pred = (p_te >= thr).astype(int)
        r = evaluate(y_te, pred, f'XLM-R base (emoji {emoji_mode})'); r.update(threshold=thr, val_macro_f1=f_va); transformer.append(r)
        print(f'XLM-R base (emoji {emoji_mode}): val F1 {f_va:.4f}  test macro-F1 {r["macro_f1"]:.4f}  acc {r["accuracy"]:.4f}  | history: {[round(h["val_macro_f1"], 4) for h in hist]}')
        if emoji_mode == 'augment':
            xlmr_pred, xlmr_prob, p_va_xlmr = pred, p_te, p_va
            xlmr_explainer = SarcasmExplainer(lambda texts, m=model, tk=tok: predict_texts(m, tk, list(texts), emoji_mode='augment', max_len=96), name='XLM-R (augment)')
        else:
            del model; torch.cuda.empty_cache()
    display(results_table(transformer))
else:
    print('skipped (RUN_TRANSFORMER=False)')
""")

# --------------------------------------------------------------------------- 6.5 stronger encoders + ensemble
md("""
## 6.5 Stronger encoders and an ensemble

Three ways to push beyond XLM-R-base, all chosen for Urdu-specific reasons:

* **MuRIL** (`google/muril-base-cased`, Khanuja et al. 2021) — a BERT encoder pre-trained on 17 Indian languages *including Urdu
  and its Roman transliteration*, so it has seen far more Urdu and Roman-Urdu than XLM-R (where Urdu is <1 % of the data).
* **mDeBERTa-v3-base** — disentangled attention + ELECTRA-style pre-training; the strongest base-size multilingual encoder on XNLI
  (run in fp32: DeBERTa-v3 overflows under fp16/bf16 autocast).
* **XLM-RoBERTa-large with LoRA** (Hu et al. 2022) — the 560M-parameter encoder would not fit a 6 GB GPU for full fine-tuning;
  rank-16 adapters on the attention/FFN projections train < 1 % of the weights and fit comfortably.
* **Ensemble** — the probability outputs of the transformers, the TF-IDF LR and the hybrid TextCNN are averaged; weights and the
  decision threshold are fitted on the *validation* split only (non-negative least squares), then applied once to the test split.
  Ensembling different inductive biases (sub-word encoder vs. char n-grams vs. hand-crafted cues) is the most reliable way to gain
  a point on a noisy-label task.

`RUN_EXTRA_MODELS` trains MuRIL and builds the ensemble (≈ 10 min on a T4 / RTX 4050); `RUN_LARGE_MODELS` adds mDeBERTa (fp32, ≈ 12 min)
and XLM-R-large/LoRA (≈ 45 min). The numbers from our full run are in §7.1.
""")
code("""
RUN_EXTRA_MODELS = RUN_TRANSFORMER
RUN_LARGE_MODELS = False
extra, val_probs, test_probs = [], {}, {}
if RUN_EXTRA_MODELS:
    from src.transformer import fine_tune, predict_texts
    todo = [('google/muril-base-cased', dict(epochs=4, lr=3e-5))]
    if RUN_LARGE_MODELS:
        todo += [('microsoft/mdeberta-v3-base', dict(epochs=4, lr=3e-5)), ('xlm-roberta-large', dict(epochs=4, lr=2e-4, lora_r=16))]
    for mname, kw in todo:
        model, tok, hist, p_va = fine_tune(mname, train[TEXT], y_tr, val[TEXT], y_va, emoji_mode='keep', max_len=96, bs=16, verbose=False, **kw)
        thr, f_va = best_threshold(y_va, p_va)
        p_te = predict_texts(model, tok, test[TEXT], emoji_mode='keep', max_len=96)
        r = evaluate(y_te, (p_te >= thr).astype(int), mname + (' (LoRA r=16)' if kw.get('lora_r') else '')); r.update(threshold=thr, val_macro_f1=f_va); extra.append(r)
        val_probs[mname], test_probs[mname] = p_va, p_te
        print(f'{r["model"]:40s} val F1 {f_va:.4f}  test macro-F1 {r["macro_f1"]:.4f}  acc {r["accuracy"]:.4f}')
        del model; torch.cuda.empty_cache()
    # ensemble with the earlier models (validation-weighted, threshold tuned on validation)
    val_probs['xlmr_base'], test_probs['xlmr_base'] = p_va_xlmr, xlmr_prob
    val_probs['tfidf_lr'], test_probs['tfidf_lr'] = predict_proba(lr_pipe, X_va), lr_prob
    val_probs['textcnn'], test_probs['textcnn'] = cnn_val_prob, cnn_test_prob
    from scipy.optimize import nnls
    names = list(val_probs); PV = np.column_stack([val_probs[n] for n in names]); PT = np.column_stack([test_probs[n] for n in names])
    w, _ = nnls(PV, y_va.astype(float)); w = w / w.sum()
    thr, f_va = best_threshold(y_va, PV @ w)
    r = evaluate(y_te, (PT @ w >= thr).astype(int), 'Ensemble (validation-weighted)'); r.update(threshold=thr, val_macro_f1=f_va); extra.append(r)
    print('ensemble weights:', dict(zip(names, np.round(w, 3))))
    print(f'Ensemble: val F1 {f_va:.4f}  test macro-F1 {r["macro_f1"]:.4f}  acc {r["accuracy"]:.4f}')
    display(results_table(extra))
""")
md("""
### 6.6 Selecting and packaging the best model
The model that is *deployed* is chosen by **validation** Macro-F1 (never by the test score). `scripts/run_ensemble.py` writes
`outputs/best_model.json` — a manifest listing the members, their weights and the decision threshold — and
`SarcasmExplainer.load_best()` reconstructs it, so `app.py` (Gradio web demo / Hugging Face Space) and the CLI always serve the
same model that the report describes. Every prediction is returned with the meaning of each emoji and the cues that fired.
""")
code("""
import json
cands = [r for r in (transformer + extra) if 'val_macro_f1' in r]
if cands:
    best = max(cands, key=lambda r: r['val_macro_f1'])
    print('best by validation Macro-F1:', best['model'], '| val', round(best['val_macro_f1'], 4), '| test macro-F1', round(best['macro_f1'], 4), '| acc', round(best['accuracy'], 4))
if os.path.exists('outputs/best_model.json'):
    man = json.load(open('outputs/best_model.json', encoding='utf-8'))
    print('deployed manifest:', man['name'], '->', [(m['tag'], m['weight']) for m in man['members']])
""")

# --------------------------------------------------------------------------- 8. summary
md("## 7. Results summary (test set, Macro-F1 is the primary metric)")
code("""
all_results = classical + neural + transformer + extra
summary = results_table(all_results); display(summary)
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
best_of = [max(g, key=lambda r: r['macro_f1']) for g in (classical, neural, transformer) if g]
for ax, r in zip(axes, best_of): plot_confusion(r, ax=ax)
plt.tight_layout(); plt.show()
""")

# results from our run (static)
summary_md = "### 7.1 Numbers obtained in our run (RTX 4050 laptop GPU)\n\n"
parts = []
for f in ["results_classical_test.csv", "results_neural_test.csv", "results_transformer_test.csv", "results_ensemble_test.csv"]:
    p = os.path.join(OUT, f)
    if os.path.exists(p):
        parts.append(pd.read_csv(p))
if parts:
    allres = pd.concat(parts).sort_values("macro_f1", ascending=False)
    allres = allres[~allres["model"].str.startswith("Single:") & (allres["macro_f1"] > 0.5)]   # drop duplicates / failed runs
    summary_md += allres[["model", "accuracy", "precision_0", "recall_0", "f1_0", "precision_1", "recall_1", "f1_1", "macro_f1"]].round(3).to_markdown(index=False)
md(summary_md)

# --------------------------------------------------------------------------- 8b. predict + explain
md("""
## 7.2 Predict and explain: the label *with the meaning of every emoji*

The detector does not stop at a 0/1 label. `SarcasmExplainer.explain()` returns, for any post, the prediction together with
(a) the meaning of each emoji in English and Urdu, its polarity, pragmatic category, irony-cue flag and position,
(b) the linguistic cues that fired (ironic-praise marker, idiom, word–emoji polarity contrast, negation, rhetorical punctuation,
textual laughter, emoji repetition, code-mixing) and (c) a one-line reading of how the emojis interact with the words.
""")
code("""
explainer = xlmr_explainer if transformer else SarcasmExplainer(lambda texts: predict_proba(lr_pipe, list(texts)), name='TF-IDF LR')
demo = ['واہ خان صاحب کیا بات ہے تبدیلی آ گئی 😂😂🙄',
        'اللہ آپ کو اپنی رحمتوں کے سائے میں رکھے اور مکمل صحت یاب فرمائے 🙏❤',
        'یہ اسمبلی بغیر ڈیزل کے چل رہی ہے کیونکہ اِسکو گدھا چلا رہا ھے 🤣😂😂😂',
        'انساں کو تھکا دیتا ہے سوچوں کا سفر بھی ... 🍁🥀',
        'اب بس بھی کرو بیچارے کی پہلے ہی دو بیویاں ہیے کیا اپ نے تیسری کے لیے ٹرائ کرنا ہے 😜',
        'شکریہ آپ کا نوازش بہت 💯🙏']
for t in demo:
    print(format_explanation(explainer.explain(t))); print('-' * 100)
# the same for a sample of test posts, as a table
explainer.explain_many(test[TEXT].sample(8, random_state=1).tolist())[['text', 'label', 'label_ur', 'p_sarcastic', 'emoji_summary_ur', 'cues']]
""")

# --------------------------------------------------------------------------- 9. error analysis
md("""
## 8. Error analysis

For the best classical model and the transformer we list the most confident mistakes, with the meaning of every emoji and
automatic phenomenon tags (negation, idiom, code-mixing, Punjabi, spelling variant, rhetorical question, quoted speech …).
The error rate per phenomenon over the whole test set shows where the models struggle.
""")
code("""
ref_pred, ref_prob, ref_name = (xlmr_pred, xlmr_prob, 'XLM-R') if transformer else (lr_pred, lr_prob, 'TF-IDF LR')
err = error_table(test, ref_pred, ref_prob, k=15)
for i, row in err.iterrows():
    print(f'[{i+1}] {row["error_type"]}  (p_sarcastic={row["p_sarcastic"]:.2f})')
    print('    text    :', row[TEXT][:220]); print('    emojis  :', row['emoji_meaning']); print('    phenomena:', row['phenomena'], '\\n')
display(phenomena_summary(test, ref_pred))
""")
md(open(os.path.join(ROOT, "report", "error_commentary.md"), encoding="utf-8").read()
   if os.path.exists(os.path.join(ROOT, "report", "error_commentary.md")) else "_(commentary)_")

# --------------------------------------------------------------------------- 10. conclusion
md("""
## 9. Conclusions

* All three families land in the same band (≈ 0.79–0.81 Macro-F1 on test; TF-IDF LR 0.811, XLM-R 0.806–0.811, TextCNN 0.799);
  the ceiling is set by **label noise**, **collection artefacts** (#طنز / تنقید) and **context dependence** (replies whose
  sarcasm only exists relative to the tweet they answer), not by model capacity.
* **Emojis are meaningful but not sufficient**: as isolated features they move Macro-F1 by < 1 point for n-gram models and
  ~1 point for the CNN, yet they are decisive for short posts, and the emoji **meaning / category / polarity** representation
  lets the models generalise across the ~540 distinct emojis, 60 % of which are rare. For XLM-R, raw emoji, Urdu meaning and
  symbol+meaning inputs are statistically tied (0.811 / 0.809 / 0.806). Hearts/flowers are the strongest *non*-sarcasm cue
  in this corpus; 😅 🤔 😉 😁 lean sarcastic.
* **Adding emoji meanings pays off in explainability**: `SarcasmExplainer` returns every prediction with the meaning of each
  emoji (English + Urdu), its polarity/category and the cues that fired, so a false positive such as literal criticism
  (تنقید) or a false negative such as a sincere-looking 😊 reply can be diagnosed directly.
* **Urdu-specific normalisation** (code points, ھ/ہ, hashtag reconstruction) is a cheap, consistent gain; aggressive
  punctuation/laughter normalisation is not, because rhetorical `؟؟؟` and ہاہاہا are cues.
* **Hand-crafted phrase features** (ironic praise, idioms, polarity contrast) reach 0.75 Macro-F1 alone and complement
  n-grams — a useful, interpretable layer for Urdu where pretrained resources are scarce.
* Future work: thread context (parent tweet), a cleaned re-annotation of conflicting posts, and Urdu-specific
  pretrained encoders (e.g. MuRIL) with longer training.

## Acknowledgements
Dataset provided by the course. Pretrained model: `xlm-roberta-base` (Conneau et al., 2020) via Hugging Face.
Libraries: scikit-learn, PyTorch, transformers, `emoji`. Emoji lexicon, Urdu idiom list and all code written for this
assignment with AI assistance (Claude); design decisions, runs and analysis reviewed by the author.
""")

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
os.makedirs(os.path.dirname(NB_PATH), exist_ok=True)
nbf.write(nb, NB_PATH)
print("wrote", NB_PATH, "with", len(cells), "cells")
