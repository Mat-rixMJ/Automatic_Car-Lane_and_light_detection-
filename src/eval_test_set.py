"""Run the validation pass over the SELECTED test clips and collect the numbers.

This ties the two halves together: select_test_videos.py picks a diverse set and
writes output/test_video_manifest.json; this runs validate_pipeline.validate on
each and prints one comparable table. Point it at the manifest so a before/after
comparison (e.g. old vs new sign detector) uses the same clips every time.

ponytail: reuses validate_pipeline.validate rather than re-implementing metrics,
so there's one source of truth for what "detected" means. Ceiling — it captures
stdout to pull the summary; if validate's print format changes, the scrape does
too. Upgrade path = have validate() return a dict. Kept as-is: one function,
no refactor of the working validator.

  python src/eval_test_set.py                 # all selected clips, 10s each
  python src/eval_test_set.py --seconds 8 --limit 5
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import PROJECT_ROOT

MANIFEST = PROJECT_ROOT / "output" / "test_video_manifest.json"


def main(seconds, limit):
    if not MANIFEST.exists():
        print(f"No manifest at {MANIFEST}")
        print("Run first: python src/select_test_videos.py --n 12")
        return

    import validate_pipeline

    clips = json.loads(MANIFEST.read_text())["selected"]
    if limit:
        clips = clips[:limit]

    print(f"Evaluating {len(clips)} selected clips ({seconds}s each)\n")
    for i, c in enumerate(clips, 1):
        path = c["path"]
        if not Path(path).exists():
            print(f"[{i}/{len(clips)}] MISSING {c['name']} — skipping")
            continue
        print(f"\n{'#'*70}\n# [{i}/{len(clips)}] {c['name']}  "
              f"(bright={c.get('brightness','?')}, veh={c.get('vehicles','?')}, "
              f"lights={c.get('lights','?')})\n{'#'*70}")
        try:
            validate_pipeline.validate(path, seconds, save_samples=False)
        except Exception as e:
            print(f"  validation failed on {c['name']}: {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Validate over the selected test clips")
    p.add_argument("--seconds", type=int, default=10)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    main(a.seconds, a.limit)
