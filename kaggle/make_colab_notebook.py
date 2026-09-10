"""Build the Colab traffic-light notebook (.ipynb) with cells split for the
Google Colab VS Code extension. Token is passed via env vars (more reliable in
the editor's notebook view than the files.upload() widget).

Run: python kaggle/make_colab_notebook.py
Then open kaggle/train_light_state_colab.ipynb, connect to Colab (T4 GPU), Run All.
"""
import json
import uuid
from pathlib import Path

HERE = Path(__file__).parent
BODY = (HERE / "train_light_state_colab.py").read_text(encoding="utf-8")
# strip the module docstring; keep the code body
doc_start = BODY.index('"""')
doc_end = BODY.index('"""', doc_start + 3) + 3
body = BODY[doc_end:].lstrip("\n")


def cell(kind, source):
    c = {"cell_type": kind, "metadata": {}, "id": uuid.uuid4().hex[:8],
         "source": source.splitlines(keepends=True)}
    if kind == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c


intro = (
    "# CarLaneI — Traffic-Light State Detector (Google Colab, 1×T4)\n\n"
    "Runs in parallel with the Kaggle IDD run + the local sign run.\n\n"
    "**Before Run All:** Runtime → Change runtime type → **T4 GPU**.\n\n"
    "Fill your Kaggle username+key in the CREDENTIALS cell (from "
    "`C:\\Users\\<you>\\.kaggle\\kaggle.json`). Optional: mount Drive so the "
    "model survives a disconnect.\n\n"
    "Output → `/content/light_state.pt` (+ Drive copy if mounted).\n"
)

gpu_check = (
    "# GPU check + install. If this prints no GPU, set Runtime -> T4 GPU.\n"
    "!nvidia-smi -L\n"
    "!pip -q install ultralytics==8.4.120 kaggle"
)

creds = (
    "# --- Kaggle credentials (needed to download the Bosch dataset) ---\n"
    "# Paste the values from your kaggle.json here:\n"
    "import os\n"
    "os.environ['KAGGLE_USERNAME'] = 'matrixmj'   # <-- your kaggle username\n"
    "os.environ['KAGGLE_KEY'] = 'PASTE_YOUR_KEY_HERE'  # <-- your kaggle key\n"
    "print('kaggle creds set for user:', os.environ['KAGGLE_USERNAME'])"
)

drive = (
    "# OPTIONAL: mount Google Drive so light_state.pt survives the session wipe.\n"
    "# Comment out this cell if you don't want Drive.\n"
    "from google.colab import drive\n"
    "drive.mount('/content/drive')"
)

cells = [
    cell("markdown", intro),
    cell("code", gpu_check),
    cell("code", creds),
    cell("code", drive),
    cell("code", body),
]

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

out = HERE / "train_light_state_colab.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, "with", len(cells), "cells")
