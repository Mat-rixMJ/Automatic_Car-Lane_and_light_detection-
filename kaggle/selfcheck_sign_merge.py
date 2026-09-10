"""Local self-check for the merged sign notebook logic (no GPU, no Kaggle).

Validates the two riskiest pieces before a Kaggle session is spent:
  1. German GTSDB super-class mapping is total over 0..42 and only emits 0..3.
  2. Indian keyword class->superclass mapping produces sane buckets on a realistic
     set of Indian sign names (mirrors the well-known 85-class Indian sign taxonomy).

Run: python kaggle/selfcheck_sign_merge.py
"""
import sys
from pathlib import Path

# import the mapping functions from the training script without running the
# Kaggle-only download/build code (guard those by __main__ there is not needed;
# they run at import). So we re-declare by exec of just the function defs.
SRC = (Path(__file__).parent / "train_sign_detector_kaggle.py").read_text(encoding="utf-8")

# Execute only up to the "ensure datasets available" section (before any os.system
# / Kaggle calls). We split on the marker comment.
MARKER = "# ------------------------------------------------- ensure datasets available"
head = SRC.split(MARKER)[0]
ns = {}
exec(compile(head, "train_sign_detector_kaggle.py", "exec"), ns)

gtsdb_super = ns["gtsdb_super"]
indian_super = ns["indian_super"]
SUPER = ns["SUPERCLASS_NAMES"]
FINE_NAMES = ns["FINE_NAMES"]
FINE_INDEX = ns["FINE_INDEX"]


def check_german():
    got = {gtsdb_super(i) for i in range(43)}
    assert got == {0, 1, 2, 3}, got
    assert gtsdb_super(FINE_INDEX["Stop"]) == 3
    assert gtsdb_super(FINE_INDEX["Speed limit (50km/h)"]) == 0
    assert gtsdb_super(FINE_INDEX["Turn right ahead"]) == 1
    assert gtsdb_super(FINE_INDEX["Road work"]) == 2
    print("GERMAN mapping OK: total over 0..42, emits only 0..3, spot-checks pass")


# A realistic Indian traffic-sign taxonomy (subset of the common 85-class set).
# expected super per our shape/colour grammar.
INDIAN_SAMPLE = {
    "Speed limit 40": "prohibitory",
    "Speed limit 60": "prohibitory",
    "No Entry": "prohibitory",
    "No Parking": "prohibitory",
    "No Stopping": "prohibitory",
    "Overtaking Prohibited": "prohibitory",
    "Horn Prohibited": "prohibitory",
    "U Turn Prohibited": "prohibitory",
    "Axle Load Limit": "prohibitory",
    "Width Limit": "prohibitory",
    "Height Limit": "prohibitory",
    "Compulsory Ahead": "mandatory",
    "Compulsory Turn Right": "mandatory",
    "Compulsory Turn Left": "mandatory",
    "Compulsory Keep Left": "mandatory",
    "Compulsory Sound Horn": "mandatory",
    "Compulsory Cycle Track": "mandatory",
    "Pedestrian Crossing": "danger",
    "School Ahead": "danger",
    "Cattle": "danger",
    "Narrow Road Ahead": "danger",
    "Road Hump": "danger",
    "Slippery Road": "danger",
    "Dangerous Dip": "danger",
    "Right Hair Pin Bend": "danger",
    "Men at Work": "danger",
    "Cycle Crossing": "danger",
    "Falling Rocks": "danger",
    # 'other' bucket - priority/informatory/unmatched
    "Stop": "other",
    "Give Way": "other",
    "Parking": "other",
    "Petrol Pump": "other",
    "Hospital": "other",
    "Round About": "danger",   # 'round about' warns of an upcoming circle -> danger
}


def check_indian():
    name_to_id = {n: i for i, n in enumerate(SUPER)}
    wrong = []
    for name, expect in INDIAN_SAMPLE.items():
        got = SUPER[indian_super(name)]
        tag = "ok " if got == expect else "MISS"
        if got != expect:
            wrong.append((name, expect, got))
        print(f"  [{tag}] {name:28s} -> {got:12s} (expected {expect})")
    # We tolerate a few 'other' fallbacks (honest default) but flag hard misclassifications
    hard = [w for w in wrong if w[2] not in ("other",) and w[1] not in ("other",)]
    print(f"\nIndian mapping: {len(INDIAN_SAMPLE)-len(wrong)}/{len(INDIAN_SAMPLE)} exact, "
          f"{len(wrong)} differ ({len(hard)} hard cross-bucket misses)")
    # a hard miss = mapped into the WRONG specific bucket (e.g. danger->prohibitory)
    assert len(hard) == 0, f"hard cross-bucket misclassifications: {hard}"


if __name__ == "__main__":
    check_german()
    print()
    check_indian()
    print("\nself-check complete.")
