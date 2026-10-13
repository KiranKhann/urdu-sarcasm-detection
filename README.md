# Urdu Sarcasm Detection (DS5007 NLP, Programming Assignment 1)

Binary classification of Urdu social-media posts as **sarcastic (1)** or **non-sarcastic (0)**,
with explicit modelling of **emoji meaning** (the dataset has emojis in 83 % of posts).

```
├── data/urdu_sarcastic_dataset.csv        # provided dataset (~20k rows)
├── notebooks/Urdu_Sarcasm_Detection.ipynb # the assignment notebook (Colab-ready, self-contained)
├── src/
│   ├── emoji_semantics.py   # emoji lexicon: meaning (EN/UR), polarity, pragmatic category, irony cue;
│   │                        # emoji->token rewriting; dense emoji features; data-driven emoji prior
│   ├── preprocessing.py     # Urdu character normalisation, social-media normalisation, tokenizer
│   ├── features.py          # hand-crafted linguistic features (idioms, ironic-praise markers,
│   │                        # word/emoji polarity contrast, negation, code-mixing ...)
│   ├── data.py              # cleaning (missing, duplicates, label conflicts) + stratified split
│   ├── baselines.py         # TF-IDF word/char n-gram models + ablation runner
│   ├── neural.py            # PyTorch TextCNN and BiLSTM+attention (+ hybrid with dense features)
│   ├── transformer.py       # XLM-R / MuRIL / mBERT fine-tuning loop
│   ├── predict.py           # predict + explain: label with the meaning of every emoji and the cues
│   └── evaluate.py          # metrics, confusion matrices, linguistic error analysis
├── scripts/run_classical.py # EDA + baselines + ablations  -> outputs/
├── scripts/run_neural.py    # neural models                -> outputs/
├── scripts/run_transformer.py [model] [emoji_mode] [epochs]
├── scripts/run_cv_ablation.py    # 5-fold CV ablation
├── scripts/make_emoji_dictionary.py, make_figures.py, build_notebook.py, build_report.py
├── outputs/                 # results tables, predictions, figures, error tables, saved models
└── report/                  # 2-3 page PDF report
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/run_classical.py
python scripts/run_neural.py
python scripts/run_transformer.py xlm-roberta-base augment 3  # emoji + its Urdu meaning (default)
python scripts/run_transformer.py xlm-roberta-base keep 3     # raw emoji only
python scripts/run_transformer.py xlm-roberta-base urdu 3     # emoji replaced by Urdu meaning
python scripts/run_cv_ablation.py
```

Or open `notebooks/Urdu_Sarcasm_Detection.ipynb` in Google Colab (GPU runtime), upload the CSV
when prompted and run all cells.

## How emojis are understood

`src/emoji_semantics.py` holds a curated lexicon of ~185 emojis. Each entry has an English
name, an Urdu gloss, a sentiment polarity, a pragmatic category (LAUGH, TEASE, SMIRK, THINK,
LOVE, SAD, ANGER, PRAYER, PRAISE, FLOWER, SHY, GESTURE, INSULT) and an *irony-cue* flag.
The lexicon drives:

* **token rewriting** for TF-IDF models (`😂` → `EMO_face_with_tears_of_joy EMOCAT_LAUGH EMOPOL_POS EMOCUE`);
* an **Urdu-gloss mode** for transformers (`😂` → `ہنسی کے آنسو`);
* **dense features**: counts per category, polarity mix, position (start/middle/end),
  longest repeat run, and the **word–emoji polarity contrast** (positive words + negative/ironic
  emoji is the classic sarcasm pattern);
* an **augment mode** (default for the transformer) that keeps the emoji *and* appends its Urdu meaning
  (`😂` → `😂 ہنسی کے آنسو`), so the model reads the symbol as used plus what it means;
* a **data-driven emoji prior** (log-odds of sarcasm per emoji, fitted on the training split only);
* an **emoji meaning dictionary** for the corpus (`outputs/emoji_meaning_dictionary.csv`: 538 emojis,
  96 % of occurrences covered by curated Urdu meanings, with P(sarcastic | emoji));
* a **predict-and-explain API** (`src/predict.py`): every prediction is returned together with the
  meaning of each emoji in the post (English + Urdu), its polarity / category, the linguistic cues
  that fired (ironic praise, idiom, polarity contrast, negation …) and a one-line reading.

```python
from src.predict import SarcasmExplainer, format_explanation
ex = SarcasmExplainer.load()                      # saved TF-IDF model (or .load_transformer(path))
print(format_explanation(ex.explain("واہ خان صاحب کیا بات ہے تبدیلی آ گئی 😂😂🙄")))
```

`explain("واہ کیا بات ہے 😂🙄")` prints a human-readable gloss of every emoji, which is used
in the error analysis.

## Acknowledgements

Dataset provided by the course instructor. Pretrained models from Hugging Face
(`xlm-roberta-base`, Conneau et al. 2020). Libraries: scikit-learn, PyTorch, transformers,
`emoji`. Code written with AI assistance (Claude), reviewed and run by the author.
