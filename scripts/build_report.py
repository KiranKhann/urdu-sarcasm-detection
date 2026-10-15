# -*- coding: utf-8 -*-
"""Build the 2-3 page PDF report from outputs/ with reportlab (Urdu rendered via arabic-reshaper + bidi)."""
import json
import os
import re

import arabic_reshaper
import pandas as pd
from bidi.algorithm import get_display
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle, KeepTogether)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs"); FIG = os.path.join(OUT, "figures"); REP = os.path.join(ROOT, "report")
os.makedirs(REP, exist_ok=True)

# fonts: Nirmala/Segoe for Latin+Urdu, Segoe UI Emoji for emojis
UR_FONT = None
for name, path in [("Tahoma", r"C:\Windows\Fonts\tahoma.ttf"), ("SegoeUI", r"C:\Windows\Fonts\segoeui.ttf"),
                   ("Arial", r"C:\Windows\Fonts\arial.ttf")]:
    if os.path.exists(path):
        try:
            pdfmetrics.registerFont(TTFont(name, path, subfontIndex=0) if path.endswith(".ttc") else TTFont(name, path))
            bold = path.replace(".ttf", "bd.ttf").replace("segoeui", "segoeuib")
            if os.path.exists(bold):
                pdfmetrics.registerFont(TTFont(name + "-Bold", bold))
                pdfmetrics.registerFontFamily(name, normal=name, bold=name + "-Bold", italic=name, boldItalic=name + "-Bold")
            UR_FONT = name; break
        except Exception:
            continue
EMO_FONT = None
if os.path.exists(r"C:\Windows\Fonts\seguiemj.ttf"):
    try:
        pdfmetrics.registerFont(TTFont("Emoji", r"C:\Windows\Fonts\seguiemj.ttf")); EMO_FONT = "Emoji"
    except Exception:
        pass

import emoji as _emoji


def ur(text: str) -> str:
    """Shape Urdu for reportlab (which has no complex-script shaping) and wrap emojis in the emoji font."""
    out = []
    last = 0
    for s in _emoji.emoji_list(text):
        out.append(_shape(text[last: s["match_start"]]))
        e = s["emoji"]
        out.append(f'<font name="{EMO_FONT}">{e}</font>' if EMO_FONT else _emoji.demojize(e))
        last = s["match_end"]
    out.append(_shape(text[last:]))
    return "".join(out)


_URDU_RUN = re.compile(r"[\u0600-\u06FF][\u0600-\u06FF ]*[\u0600-\u06FF]|[\u0600-\u06FF]")


def _shape(t):
    """Reshape + reorder ONLY the Urdu runs; Latin text and our <b>/<i> markup stay untouched.
    (The paragraph is left-to-right, so each Urdu run is rendered in visual order inline.)"""
    t = re.sub(r"&(?![a-z]+;)", "&amp;", t)
    return _URDU_RUN.sub(lambda m: get_display(arabic_reshaper.reshape(m.group(0))), t)


ss = getSampleStyleSheet()
HB = (UR_FONT + "-Bold") if UR_FONT and (UR_FONT + "-Bold") in pdfmetrics.getRegisteredFontNames() else (UR_FONT or "Helvetica-Bold")
BODY = ParagraphStyle("body", parent=ss["Normal"], fontName=UR_FONT or "Helvetica", fontSize=8.1, leading=10.1, alignment=TA_JUSTIFY, spaceAfter=2.5)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName=HB, fontSize=11.5, leading=13, spaceBefore=4, spaceAfter=2)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName=HB, fontSize=9.8, leading=11, spaceBefore=3, spaceAfter=1)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=7.2, leading=8.6)
TITLE = ParagraphStyle("title", parent=ss["Title"], fontName=HB, fontSize=14, leading=16, spaceAfter=1)


def P(t, st=BODY):
    return Paragraph(ur(t), st)


def df_table(df, col_widths=None, font=7.2):
    data = [list(df.columns)] + df.values.tolist()
    data = [[Paragraph(ur(str(c)), ParagraphStyle("c", parent=SMALL, fontSize=font, leading=font + 1.6)) for c in row] for row in data]
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDE5F0")),
                           ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                           ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    return t


summary = json.load(open(os.path.join(OUT, "classical_summary.json"), encoding="utf-8"))
eda = summary["eda"]
res = pd.concat([pd.read_csv(os.path.join(OUT, f)) for f in
                 ["results_classical_test.csv", "results_neural_test.csv", "results_transformer_test.csv", "results_ensemble_test.csv"]
                 if os.path.exists(os.path.join(OUT, f))]).sort_values("macro_f1", ascending=False)
# keep the table to one page: drop the weaker duplicates of a family
res = res[~res["model"].str.contains("emoji remove|ComplementNB|Ensemble: mean|Single:|mdeberta", regex=True)]
man = json.load(open(os.path.join(OUT, "best_model.json"), encoding="utf-8")) if os.path.exists(os.path.join(OUT, "best_model.json")) else None
cv = pd.read_csv(os.path.join(OUT, "ablations_cv.csv")) if os.path.exists(os.path.join(OUT, "ablations_cv.csv")) else None
best = res.iloc[0]

story = []
story.append(Paragraph("Urdu Sarcasm Detection with Emoji Semantics", TITLE))
story.append(P("DS5007 Natural Language Processing (Fall 2026) — Programming Assignment 1 — Kiran (kiran.poulta@gmail.com). "
               "Code, notebook and outputs: <i>notebooks/Urdu_Sarcasm_Detection.ipynb</i>, <i>src/</i>, <i>outputs/</i>.", SMALL))

story.append(Paragraph("1. Data and preparation", H1))
cc = eda["class_counts"]
story.append(P(f"The provided sheet has {eda['rows_raw']:,} rows. Cleaning removed {eda['rows_missing_dropped']} rows without text, "
               f"{eda['exact_duplicates_dropped']:,} exact duplicate posts (which would otherwise leak between train and test) and "
               f"{eda['conflicting_label_texts']} texts that occur with <b>both</b> labels ({eda['conflicting_rows_dropped']:,} rows; unlearnable annotation conflicts). "
               f"{eda['rows_clean']:,} posts remain: {cc['0']:,} non-sarcastic / {cc['1']:,} sarcastic (59/41 %) — the raw file only looked balanced because the sarcastic class was heavily duplicated. "
               f"A stratified 70/15/15 split gives {eda['split']['train']:,}/{eda['split']['val']:,}/{eda['split']['test']:,} posts; all model selection uses the validation split, the test split is used once."))
story.append(P(f"Posts are short (median {eda['words']['50%']:.0f} words, max {eda['words']['max']:.0f}); sarcastic posts are longer (median 20 vs 13 words). "
               f"{eda['share_with_emoji']:.0%} of posts contain emojis (88 % of non-sarcastic vs 67 % of sarcastic — the non-sarcastic class is dominated by poetry and greetings decorated with ❤ 🌹 💔). "
               f"{eda['share_latin_tokens']:.0%} contain Latin-script tokens (code-mixing: ISPR, PM, hashtags), {eda['share_hashtag']:.0%} hashtags (often scraped as words_with_underscores), "
               f"{eda['share_mention']:.0%} mentions, {eda['share_url']:.0%} URLs, {eda['share_quotes']:.0%} quoted speech (`` … ''). Further quality issues: Arabic/Urdu code-point variants (ك/ک, ي/ی), ھ typed for ہ (ھے), "
               "elongation (کدووووو), ~8 % Punjabi in Urdu script, and collection artefacts — some posts still carry the harvesting hashtags #طنز / #مزاح / #sarcasm, which a linear model learns as its strongest feature. "
               "Many clearly mocking political posts are labelled 0 and some plain shayari with 💔 is labelled 1, which caps the attainable accuracy for every model."))
story.append(Image(os.path.join(FIG, "eda.png"), width=16 * cm, height=4.3 * cm))

story.append(Paragraph("2. Urdu-aware preprocessing and emoji semantics", H1))
story.append(P("<b>Normalisation</b> (src/preprocessing.py) is conservative and every step is switchable for ablation: Unicode NFC and code-point unification (ك→ک, ي→ی, ة→ہ, أ→ا, Sindhi ٸ→ئ, Urdu digits→ASCII); "
               "diacritic removal; whole-word keyboard fixes (ھے→ہے, ھیں→ہیں, کوئ→کوئی — ھ is a real letter in تھا so only listed words are changed); URL→URL, @user→@USER; hashtag reconstruction "
               "(ووٹ_کو_عزت_دو → HASHTAG ووٹ کو عزت دو); laughter (ہاہاہاہا, hahaha, lol→LAUGH) and elongation (کدووووو→کدوو) normalisation; Urdu punctuation mapped to ASCII but <b>kept</b> as tokens because rhetorical ؟؟؟ and … are cues. "
               "Roman/English and Punjabi tokens are kept. Tokenisation is regex-based (Urdu word runs, Latin words, numbers, punctuation runs, single emojis). Emojis are extracted before normalisation because the zero-width joiner we strip from Urdu text is part of emoji sequences (🤦‍♂️)."))
story.append(P("<b>Emoji semantics</b> (src/emoji_semantics.py). A curated lexicon of ~150 emojis gives each an English name, an <b>Urdu gloss</b> (😂 ہنسی کے آنسو, 🙄 آنکھیں گھمانا, 💔 ٹوٹا دل), a polarity (+/0/−), a pragmatic category "
               "(LAUGH, TEASE, SMIRK, THINK, LOVE, SAD, ANGER, PRAYER, PRAISE, FLOWER, SHY, GESTURE, INSULT) and an <i>irony-cue</i> flag (😏 🙄 😜 😅 🤔 …). Skin tones and variation selectors are canonicalised (❤️=❤, ✌🏻=✌). The lexicon drives: "
               "(i) seven representations tested in the ablation — remove, keep raw, meaning token (EMO_face_with_tears_of_joy), category token (EMOCAT_LAUGH), meaning+category+polarity+cue, Urdu gloss (😂 → ہنسی کے آنسو), and <b>augment</b> (😂 → 😂 ہنسی کے آنسو: the symbol as used <i>plus</i> its meaning in the language of the post — the default input of the transformer); "
               "(ii) 27 dense emoji features: counts per category, polarity mix, position (start/middle/end), longest repeat run (😂😂😂😂), laughter-only; (iii) a data-driven emoji sarcasm prior (log-odds per emoji, fitted on the training split only); "
               "(iv) <b>word–emoji polarity contrast</b>: positive words or ironic-praise markers (واہ, ماشاءاللہ, کیا بات ہے, شاباش, تبدیلی, خان صاحب) combined with an irony-cue/negative emoji — the canonical sarcasm pattern; (v) an <b>emoji meaning dictionary</b> of the 538 emojis in the corpus (96 % of occurrences have a curated Urdu meaning) with P(sarcastic|emoji); and (vi) a <b>predict-and-explain API</b> (src/predict.py) that returns, with every label, the meaning of each emoji in English and Urdu, its polarity/category, the cues that fired (ironic praise, idiom, polarity contrast, negation, rhetorical punctuation, repetition, code-mixing) and a one-line reading, e.g. for واہ خان صاحب کیا بات ہے تبدیلی آ گئی 😂😂🙄 → <i>sarcastic (0.73)</i>; 😂 = ہنسی کے آنسو, 🙄 = آنکھیں گھمانا; cues: ironic-praise marker, irony-cue emoji, positive words + ironic emoji; reading: the praise words are a vehicle for mockery."))
story.append(P("<b>Hand-crafted Urdu phrase features</b> (bonus, src/features.py, 72 features): ironic-praise markers, 35 idioms/multi-word expressions (اپنے پیر پر کلہاڑی, چراغ تلے اندھیرا, دودھ کا راکھا بلا, ٹوپی ڈرامہ, کر لو گل …), "
               "negation words, small positive/negative lexicons, question/ellipsis/repetition counts, quoted speech, code-mixing ratio, laughter, hashtags/mentions/URLs. In the training split P(sarcastic|emoji) ranges from 0.00–0.06 for ❣ 💐 🌷 💞 ♥ 🌹 💔 to 0.55–0.61 for 😄 😆 😁 😉 🤔 😅 🤪 (base rate 0.41); 😂 alone is only mildly informative (0.50) — it marks humour, and the words decide whether the laughter is <i>at</i> someone."))

story.append(Paragraph("3. Models and experiments", H1))
story.append(P("<b>Classical:</b> TF-IDF word uni/bi-grams + character 2–5-grams (within word boundaries; robust to spelling variation and suffix morphology) + the standardised hand-crafted block, with Logistic Regression (liblinear, class-balanced), a calibrated Linear SVM and Complement NB; C tuned on validation. "
               "<b>Neural (PyTorch):</b> TextCNN (200-d embeddings learnt from scratch, filter widths 2–5 × 128, max-pooling) and BiLSTM (128/direction) + additive attention + max pooling; hybrid variants concatenate the hand-crafted vector; Adam, class-weighted CE, dropout 0.5, early stopping on validation Macro-F1, threshold tuned on validation; every emoji is one token so emoji sequences are visible. "
               "<b>Transformer:</b> xlm-roberta-base (pre-trained on 100 languages incl. Urdu) fine-tuned with the 250k×768 word-embedding matrix frozen (192M of 278M parameters — fits a 6 GB GPU and regularises on 10k examples; all 12 encoder layers and the head are trained; AdamW 2e-5, warm-up, class-weighted loss, fp16, max 96 sub-words, 3 epochs, early stopping, threshold tuned on validation), with three emoji inputs: symbol + Urdu meaning (augment), raw symbol, Urdu meaning only. Macro-F1 is the primary comparison metric."))

story.append(Paragraph("4. Results (test split, n = 2 186)", H1))
rt = res[["model", "accuracy", "precision_0", "recall_0", "f1_0", "precision_1", "recall_1", "f1_1", "macro_f1"]].round(3)
rt.columns = ["model", "acc", "P(0)", "R(0)", "F1(0)", "P(1)", "R(1)", "F1(1)", "Macro-F1"]
story.append(df_table(rt, col_widths=[6.6 * cm] + [1.3 * cm] * 8))
story.append(Spacer(1, 3))
cms = [f for f in ["cm_ensemble.png", "cm_muril-base-cased_keep.png", "cm_lr.png"] if os.path.exists(os.path.join(FIG, f))]
if cms:
    story.append(Table([[Image(os.path.join(FIG, f), width=4.6 * cm, height=4.1 * cm) for f in cms]]))
story.append(P("Confusion matrices (rows = true, cols = predicted): deployed ensemble, best single transformer (MuRIL), best classical. All families mis-classify more sarcastic posts than non-sarcastic ones (lower recall on class 1), i.e. subtle mockery without lexical markers is the hard case."))
story.append(P("<b>Ensemble and model selection.</b> The probability outputs of the transformers, the TF-IDF LR and the hybrid TextCNN are combined; member weights (non-negative least squares) or the member subset (greedy forward selection) and the decision threshold are fitted on the <i>validation</i> split only and applied once to the test split. Combining three inductive biases - sub-word encoder, character n-grams, hand-crafted emoji/phrase cues - is worth about +1.5 Macro-F1 over the best single model, the largest gain of any change in this project, because the members make partly different errors on the noisy posts. "
               + (f"The system selected by validation Macro-F1 is <b>{man['name'].replace('Ensemble: ', '')}</b> (validation {man['val_macro_f1']:.3f}; test Macro-F1 {man['test_macro_f1']:.3f}, accuracy {man['test_accuracy']:.1%}); it is the model packaged in <i>outputs/best_model.json</i> and served by <i>app.py</i>." if man else "")))

story.append(Paragraph("5. Ablation study (5-fold CV on train+val, TF-IDF LR, mean ± std Macro-F1)", H2))
if cv is not None:
    cvt = cv[["config", "macro_f1_mean", "macro_f1_std", "acc_mean"]].round(4)
    cvt.columns = ["configuration", "Macro-F1", "± std", "accuracy"]
    story.append(df_table(cvt, col_widths=[8 * cm, 2 * cm, 2 * cm, 2 * cm]))
def md_paragraphs(path):
    if not os.path.exists(path):
        return
    for para in [p for p in open(path, encoding="utf-8").read().split("\n") if p.strip()]:
        para = re.sub(r"^#+\s*", "", para)
        para = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", para)
        para = re.sub(r"(?<!\*)\*(?!\*)(.+?)\*(?!\*)", r"<i>\1</i>", para)
        para = re.sub(r"^\s*[-*]\s+", "• ", para)
        story.append(P(para))


md_paragraphs(os.path.join(REP, "ablation_commentary.md"))

story.append(Paragraph("6. Error analysis", H1))
md_paragraphs(os.path.join(REP, "error_commentary.md"))

story.append(Paragraph("7. Conclusion", H1))
story.append(P((f"The deployed system, selected on validation, is the {man['name'].replace('Ensemble: ', '')} ensemble: test Macro-F1 {man['test_macro_f1']:.3f} / accuracy {man['test_accuracy']:.1%}; the validation-weighted ensemble reaches {best['macro_f1']:.3f} / {best['accuracy']:.1%}, and the best single encoder (MuRIL, pre-trained on Urdu and Roman-Urdu) 0.813 / 81.7 %. " if man else f"The best system ({best['model']}) reaches Macro-F1 {best['macro_f1']:.3f} / accuracy {best['accuracy']:.3f}. ") + "All single-model families fall in a narrow band because the ceiling is set by label noise and context dependence rather than model capacity; only ensembling moves it. "
               "Emoji meaning matters most for short, lexically neutral posts and as a generalising representation (category/polarity) for rare emojis; hearts/flowers are the strongest non-sarcasm cue, 😅 🤔 😉 😁 lean sarcastic. Urdu code-point normalisation is a cheap consistent gain, aggressive punctuation/laughter normalisation is not. "
               "Hand-crafted phrase features (ironic praise, idioms, polarity contrast) reach 0.75 Macro-F1 alone and remain useful on top of n-grams. Future work: conversational context (parent tweet), re-annotation of conflicting posts, and Urdu-specific encoders (MuRIL) with longer training."))
story.append(P("<b>Acknowledgements.</b> Dataset provided by the course. xlm-roberta-base (Conneau et al., 2020) via Hugging Face; scikit-learn, PyTorch, transformers, emoji. Code and lexicons written with AI assistance (Claude); all experiments run and reviewed by the author.", SMALL))

doc = SimpleDocTemplate(os.path.join(REP, "Urdu_Sarcasm_Report.pdf"), pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.1 * cm, bottomMargin=1.1 * cm,
                        title="Urdu Sarcasm Detection with Emoji Semantics", author="Kiran")
doc.build(story)
print("wrote", os.path.join(REP, "Urdu_Sarcasm_Report.pdf"))
