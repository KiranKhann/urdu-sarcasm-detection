# -*- coding: utf-8 -*-
"""Zip the transformer checkpoints referenced by outputs/best_model.json into outputs/release/
(one zip per member, stored uncompressed - safetensors do not compress) for the GitHub release."""
import json
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs"); REL = os.path.join(OUT, "release")
os.makedirs(REL, exist_ok=True)
man = json.load(open(os.path.join(OUT, "best_model.json"), encoding="utf-8"))
made = []
for m in man["members"]:
    if m["kind"] != "transformer":
        continue
    src = os.path.join(ROOT, m["path"])
    dst = os.path.join(REL, os.path.basename(src) + ".zip")
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as z:
        for dp, _, fns in os.walk(src):
            for fn in fns:
                full = os.path.join(dp, fn)
                z.write(full, os.path.relpath(full, OUT))      # unzips to outputs/model_<tag>/...
    made.append((dst, round(os.path.getsize(dst) / 1e6)))
print("release assets:", made)
