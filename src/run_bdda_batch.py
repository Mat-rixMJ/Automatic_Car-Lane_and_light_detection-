"""Run the pipeline on multiple BDDA test videos sequentially with live display.

Produces one combined output video and prints per-clip stats.
Usage: python run_bdda_batch.py --n 10
"""

import sys
import time
import argparse
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))
from run_pipeline_fast import run_pipeline
from utils import PROJECT_ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10, help="Number of BDDA clips to process")
    p.add_argument("--split", default="test", choices=["test", "training"])
    args = p.parse_args()

    vid_dir = PROJECT_ROOT / "BDDA" / args.split / "camera_videos"
    vids = sorted(vid_dir.glob("*.mp4"))[:args.n]
    print(f"Running pipeline on {len(vids)} BDDA {args.split} clips (live + record)\n")

    out_path = PROJECT_ROOT / "output" / f"bdda_{args.split}_{args.n}clips.mp4"

    # Get frame size from first clip
    cap = cv2.VideoCapture(str(vids[0]))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    for i, vid in enumerate(vids, 1):
        print(f"\n{'='*50}")
        print(f"[{i}/{len(vids)}] {vid.name}")
        print(f"{'='*50}")
        clip_out = PROJECT_ROOT / "output" / f"bdda_clip_{vid.stem}.mp4"
        run_pipeline(str(vid), str(clip_out), crop_center=False, live=True)

    print(f"\n\nAll {len(vids)} clips processed. Individual outputs in output/bdda_clip_*.mp4")


if __name__ == "__main__":
    main()
