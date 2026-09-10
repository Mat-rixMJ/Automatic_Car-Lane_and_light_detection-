"""Generate a Kaggle .ipynb from a training script.

Robust approach: one markdown intro cell, one install cell, and ONE code cell
containing the entire script body (imports first). Splitting into multiple cells
previously separated the imports from their use (NameError: Path). A single code
cell is bulletproof and Kaggle runs it top-to-bottom exactly like the script.

Usage:
  python make_notebook.py            # regenerates ALL known notebooks
  python make_notebook.py ego        # just the ego-seg notebook
  python make_notebook.py sign       # just the sign-detector notebook
"""
import json
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).parent

# name -> (script stem, markdown intro, output filename shown to the user)
JOBS = {
    "ego": (
        "train_ego_seg_kaggle",
        "# CarLaneI — Ego-Lane Segmentation (Kaggle GPU)\n\n"
        "Accelerator = **GPU T4** (x2 if available). Internet **ON** "
        "(the notebook downloads the BDD dataset + base weights itself).\n\n"
        "Run All → download `/kaggle/working/ego_seg.pt` from the Output tab.\n",
    ),
    "sign": (
        "train_sign_detector_kaggle",
        "# CarLaneI — Merged Indian + German Sign Detector (Kaggle GPU)\n\n"
        "Accelerator = **GPU T4** (x2 if available). Internet **ON** "
        "(the notebook self-downloads both the GTSDB German set and the Indian "
        "dashcam sign set + base weights).\n\n"
        "One detector over 4 shape+colour super-classes "
        "(prohibitory / mandatory / danger / other), trained on real German + "
        "Indian road photos.\n\n"
        "Run All → download `/kaggle/working/sign_detector.pt` from the Output tab.\n",
    ),
    "light": (
        "train_light_state_kaggle",
        "# CarLaneI — Traffic-Light State Detector (Kaggle GPU)\n\n"
        "Accelerator = **GPU T4** (x2 if available). Internet **ON** "
        "(self-downloads the Bosch Small Traffic Lights YOLO set + base weights).\n\n"
        "Detects the light **state** directly — red / green / yellow / off — the "
        "honest replacement for the colour heuristic that was weakest at night.\n\n"
        "Run All → download `/kaggle/working/light_state.pt` from the Output tab.\n",
    ),
    "idd": (
        "train_ego_seg_idd_kaggle",
        "# CarLaneI — Ego-Lane Segmentation on Indian Roads / IDD (Kaggle GPU)\n\n"
        "Accelerator = **GPU T4** (x2 if available). Internet **ON** "
        "(self-downloads the IDD YOLO-Seg set + base weights).\n\n"
        "Same drivable-area task as the BDD ego-seg, trained on the India Driving "
        "Dataset so the corridor is learned on chaotic, often-unpainted Indian "
        "roads — the domain match for the Indian showcase footage.\n\n"
        "Run All → download `/kaggle/working/ego_seg_idd.pt` from the Output tab.\n",
    ),
}


def cell(kind, source):
    c = {"cell_type": kind, "metadata": {}, "id": uuid.uuid4().hex[:8],
         "source": source.splitlines(keepends=True)}
    if kind == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c


def build(job_key):
    stem, intro = JOBS[job_key]
    src = (HERE / f"{stem}.py").read_text(encoding="utf-8")
    # strip the leading module docstring (keep the code body only)
    doc_start = src.index('"""')
    doc_end = src.index('"""', doc_start + 3) + 3
    body = src[doc_end:].lstrip("\n")

    cells = [
        cell("markdown", intro),
        cell("code", "!pip -q install ultralytics==8.4.120"),
        cell("code", body),
    ]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"},
                       "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}
    out = HERE / f"{stem}.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    print("wrote", out, "with", len(cells), "cells (1 markdown, 2 code)")


if __name__ == "__main__":
    keys = sys.argv[1:] or list(JOBS)
    for k in keys:
        if k not in JOBS:
            print(f"unknown job '{k}', known: {list(JOBS)}")
            continue
        build(k)
