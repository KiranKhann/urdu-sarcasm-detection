# -*- coding: utf-8 -*-
"""
Web demo: Urdu sarcasm detection with emoji meanings.

    python app.py                 # local Gradio UI at http://127.0.0.1:7860
    python app.py --cli "متن"     # one-off prediction in the terminal

The model is whatever outputs/best_model.json points to (written by scripts/run_ensemble.py):
a single fine-tuned encoder or a validation-weighted ensemble.  If a member is missing locally
(e.g. a transformer checkpoint not included in the download) the app falls back to the saved
TF-IDF + Logistic-Regression model so it always runs.
This file is also a ready-made Hugging Face Space (Gradio SDK) entry point.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.predict import SarcasmExplainer, format_explanation  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_model():
    try:
        ex = SarcasmExplainer.load_best(os.path.join(ROOT, "outputs", "best_model.json"))
        note = f"model: {ex.name} (threshold {ex.threshold:.2f}, selected by validation Macro-F1)"
    except Exception as e:  # missing checkpoint etc. -> always-available fallback
        ex = SarcasmExplainer.load(os.path.join(ROOT, "outputs", "model_tfidf_lr.joblib"))
        note = f"model: {ex.name} (fallback: {type(e).__name__}: {str(e)[:120]})"
    return ex, note


EXAMPLES = [
    "واہ خان صاحب کیا بات ہے تبدیلی آ گئی 😂😂🙄",
    "اللہ آپ کو اپنی رحمتوں کے سائے میں رکھے اور مکمل صحت یاب فرمائے 🙏❤",
    "نیا پاکستان مبارک ہو، پٹرول دو سو روپے لیٹر 🤣🤣",
    "میری ماں دنیا کی بہترین انسان ہیں 🥰",
    "ماشاءاللہ کیا ٹیم ہے، پانچ میچ لگاتار ہار گئی 🙄🙄",
    "آج موسم بہت اچھا ہے، سب دوستوں کے ساتھ چائے پی 🌹☕",
]


def render(result):
    """Markdown block: label, probability, emoji meanings table, cues, reading."""
    lab = "🟠 **Sarcastic / طنزیہ**" if result["label"] == "sarcastic" else "🟢 **Non-sarcastic / غیر طنزیہ**"
    out = [f"### {lab}", f"P(sarcastic) = **{result['p_sarcastic']:.2f}**  ·  {result['model']}", ""]
    if result["emojis"]:
        out += ["| emoji | meaning (English) | معنی (اردو) | category | polarity | irony cue | position |", "|---|---|---|---|---|---|---|"]
        seen = set()
        for e in result["emojis"]:
            key = (e["emoji"], e["position"])
            if key in seen:
                continue
            seen.add(key)
            out.append(f"| {e['emoji']} | {e['meaning_en']} | {e['meaning_ur']} | {e['category']} | {e['polarity']:+d} | {'yes' if e['irony_cue'] else 'no'} | {e['position']} |")
    else:
        out.append("_no emoji in this post_")
    out += ["", "**Cues:** " + ("; ".join(result["cues"]) if result["cues"] else "none"),
            "", "**Reading:** " + result["reading"]]
    return "\n".join(out)


def main():
    ex, note = load_model()
    if len(sys.argv) > 2 and sys.argv[1] == "--cli":
        print(note); print(format_explanation(ex.explain(" ".join(sys.argv[2:]))))
        return
    import gradio as gr

    def predict(text):
        text = (text or "").strip()
        if not text:
            return "_enter an Urdu post_"
        return render(ex.explain(text))

    with gr.Blocks(title="Urdu Sarcasm Detector") as demo:
        gr.Markdown("# Urdu sarcasm detection with emoji meanings\n"
                    "Paste an Urdu social-media post. The detector returns the label, the probability, "
                    "**what every emoji means** (English + Urdu, polarity, pragmatic category, irony cue) and the "
                    "linguistic cues that fired.\n\n" + note)
        with gr.Row():
            inp = gr.Textbox(label="Urdu post", lines=3, rtl=True, placeholder="یہاں اردو متن لکھیں ...")
        btn = gr.Button("Detect", variant="primary")
        out = gr.Markdown()
        btn.click(predict, inp, out)
        inp.submit(predict, inp, out)
        gr.Examples(EXAMPLES, inputs=inp, outputs=out, fn=predict, cache_examples=False)
    demo.launch()


if __name__ == "__main__":
    main()
