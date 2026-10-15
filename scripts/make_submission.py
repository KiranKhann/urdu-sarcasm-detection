# -*- coding: utf-8 -*-
"""Assemble the submission bundle:  submission/  +  submission.zip

  Urdu_Sarcasm_Detection.ipynb   executed notebook (clean, runnable; Colab-ready)
  Urdu_Sarcasm_Detection.py      the same code exported as a .py script
  Urdu_Sarcasm_Detection.html    rendered notebook with all outputs (for reading without Jupyter)
  Urdu_Sarcasm_Report.pdf        2-3 page report
  results/                       results tables (per-class P/R/F1, Macro-F1), confusion matrices, error tables,
                                 ablations, emoji meaning dictionary
  models/                        saved TF-IDF vectoriser+LR pipeline and BiLSTM weights (transformers: see README)
  src/, scripts/, data/          reproducible code and the dataset
  README.md                      how to run + requirement-to-location map + acknowledgements
"""
import glob
import os
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = os.path.join(ROOT, "submission")
NB = os.path.join(ROOT, "notebooks", "Urdu_Sarcasm_Detection.ipynb")
shutil.rmtree(SUB, ignore_errors=True)
for d in ["results", "results/figures", "models", "src", "scripts", "data"]:
    os.makedirs(os.path.join(SUB, d), exist_ok=True)

shutil.copy(NB, os.path.join(SUB, "Urdu_Sarcasm_Detection.ipynb"))
shutil.copy(os.path.join(ROOT, "report", "Urdu_Sarcasm_Report.pdf"), SUB)
for fmt in ["script", "html"]:
    subprocess.check_call([sys.executable, "-m", "nbconvert", "--to", fmt, "--output-dir", SUB, NB],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
# make the exported script plain Python: the %%writefile cells are redundant (src/ is shipped next to it)
# and display() is a Jupyter builtin
import re
spath = os.path.join(SUB, "Urdu_Sarcasm_Detection.py")
lines = open(spath, encoding="utf-8").read().split("\n")
lines = [l for l in lines if not l.startswith("get_ipython().run_cell_magic('writefile'")]
script = "\n".join(lines).replace("display(", "print(")
script = script.replace("#!/usr/bin/env python\n# coding: utf-8\n",
                        "#!/usr/bin/env python\n# coding: utf-8\n# Exported from Urdu_Sarcasm_Detection.ipynb. Run from the folder that contains src/ and data/.\n"
                        "import matplotlib; matplotlib.use('Agg')\n")
open(spath, "w", encoding="utf-8").write(script)
assert "get_ipython" not in script, "notebook magics left in the exported script"

OUT = os.path.join(ROOT, "outputs")
for pat in ["results_*.csv", "ablations_*.csv", "errors_*.csv", "phenomena_*.csv", "emoji_meaning_dictionary.csv",
            "emoji_sarcasm_rate.csv", "top_features_lr.json", "classical_summary.json", "neural_summary.json",
            "ensemble_summary.json", "transformer_*.json", "pred_test_*.csv"]:
    for f in glob.glob(os.path.join(OUT, pat)):
        shutil.copy(f, os.path.join(SUB, "results"))
for f in glob.glob(os.path.join(OUT, "figures", "*.png")):
    shutil.copy(f, os.path.join(SUB, "results", "figures"))
for f in ["model_tfidf_lr.joblib", "model_bilstm_hybrid.pt", "model_textcnn_hybrid.pt", "best_model.json"]:
    if os.path.exists(os.path.join(OUT, f)):
        shutil.copy(os.path.join(OUT, f), os.path.join(SUB, "models"))
shutil.copy(os.path.join(ROOT, "app.py"), SUB)
for f in glob.glob(os.path.join(ROOT, "src", "*.py")):
    shutil.copy(f, os.path.join(SUB, "src"))
for f in glob.glob(os.path.join(ROOT, "scripts", "*.py")):
    shutil.copy(f, os.path.join(SUB, "scripts"))
shutil.copy(os.path.join(ROOT, "data", "urdu_sarcastic_dataset.csv"), os.path.join(SUB, "data"))
shutil.copy(os.path.join(ROOT, "requirements.txt"), SUB)
shutil.copy(os.path.join(ROOT, "SUBMISSION_README.md"), os.path.join(SUB, "README.md"))

zpath = os.path.join(ROOT, "submission.zip")
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for dp, _, fns in os.walk(SUB):
        for fn in fns:
            full = os.path.join(dp, fn)
            z.write(full, os.path.relpath(full, ROOT))
print("submission/ written;", zpath, f"{os.path.getsize(zpath) / 1e6:.1f} MB")
