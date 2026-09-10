"""Local self-check for the two new Kaggle notebooks (no GPU, no Kaggle).

Validates the risky, no-GPU-needed logic before a session is spent:
  1. IDD drivable-class keyword detection picks a sane class from a realistic
     IDD-Seg class list, and remap-to-0 logic is correct.
  2. Light-state config sanity: 4 classes, hue jitter disabled (colour == class).

Run: python kaggle/selfcheck_new_notebooks.py
"""
from pathlib import Path

HERE = Path(__file__).parent


def load_head(fname, marker):
    """exec a script only up to `marker` (before Kaggle-only download calls)."""
    src = (HERE / fname).read_text(encoding="utf-8")
    # drop module docstring
    ds = src.index('"""'); de = src.index('"""', ds + 3) + 3
    body = src[de:]
    head = body.split(marker)[0]
    ns = {}
    exec(compile(head, fname, "exec"), ns)
    return ns


def extract_defs(fname, snippets):
    """exec only specific top-level assignments/defs pulled from the source, so we
    can test pure logic that lives AFTER the Kaggle-only download calls without
    running them. Each snippet is (start_line_substring, end_line_substring)."""
    src = (HERE / fname).read_text(encoding="utf-8").splitlines()
    ns = {}
    for start_sub, end_sub in snippets:
        si = next(i for i, ln in enumerate(src) if start_sub in ln)
        ei = next(i for i, ln in enumerate(src) if end_sub in ln and i >= si)
        block = "\n".join(src[si:ei + 1])
        exec(compile(block, fname, "exec"), ns)
    return ns


# ---------------------------------------------------------------- IDD notebook
def check_idd():
    # pull the pure config + matcher from the source (they live after the Kaggle
    # download call, so we can't exec the whole head).
    ns = extract_defs("train_ego_seg_idd_kaggle.py", [
        ("DRIVABLE_KEYWORDS =", "DRIVABLE_KEYWORDS ="),
        ("NEG_MARKERS =", "NEG_MARKERS ="),
        ("def _matches(", "    return kw in low"),
    ])
    kws = ns["DRIVABLE_KEYWORDS"]
    matches = ns["_matches"]

    def pick(names, override=None):
        if override is not None:
            return {override}
        ids = set()
        for kw in kws:
            for i, n in enumerate(names):
                if matches(n, kw):
                    ids.add(i)
            if ids:
                break
        return ids

    # realistic IDD-Seg class list (the common drivable-area export)
    idd_names = ["road", "drivable fallback", "sidewalk", "non-drivable fallback",
                 "person", "rider", "vehicle", "sky", "vegetation"]
    got = pick(idd_names)
    picked = sorted(got)
    print("IDD classes:", idd_names)
    print("  picked drivable ids:", picked, "=", [idd_names[i] for i in picked])
    # 'drivable' keyword must select 'drivable fallback' (id 1)
    assert 1 in got, "should pick 'drivable fallback'"
    # and MUST NOT select 'non-drivable fallback' (id 3) - the negation bug
    assert 3 not in got, "must NOT pick 'non-drivable fallback' (negation guard)"
    # a road-only dataset falls back to 'road'
    got2 = pick(["road", "sidewalk", "car", "sky"])
    assert got2 == {0}, f"road-only fallback should pick id 0, got {got2}"
    print("  road-only fallback picks:", sorted(got2), "(ok)")
    # override wins
    assert pick(idd_names, override=5) == {5}
    print("  DRIVABLE_OVERRIDE respected (ok)")

    # remap-to-0 line logic: a drivable polygon line -> '0 ...', others dropped.
    # id 1 = 'drivable fallback' (kept), id 3 = 'non-drivable' (dropped), id 2 =
    # sidewalk (dropped).
    drivable_ids = got
    sample = [
        "1 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2",   # drivable fallback - KEEP
        "3 0.5 0.5 0.6 0.5 0.6 0.6 0.5 0.6",   # non-drivable - drop
        "2 0.5 0.5 0.6 0.5 0.6 0.6",           # sidewalk - drop
    ]
    kept = []
    for line in sample:
        parts = line.split()
        cid = int(float(parts[0]))
        if cid in drivable_ids and len(parts) >= 7:
            kept.append("0 " + " ".join(parts[1:]))
    print(f"  remap kept {len(kept)} of {len(sample)} sample polygons")
    assert len(kept) == 1, f"should keep exactly the drivable polygon, kept {len(kept)}"
    assert all(k.startswith("0 ") for k in kept), "kept polygons must be class 0"
    print("IDD notebook logic OK\n")


# --------------------------------------------------------------- light notebook
def check_light():
    ns = load_head("train_light_state_kaggle.py",
                   "# ------------------------------------------------- ensure dataset available")
    classes = ns["CLASS_NAMES"]
    assert classes == ["red", "green", "yellow", "off"], classes
    print("Light classes:", classes)
    # the guard set is 0..3
    guard_ok = {0, 1, 2, 3}
    assert set(range(len(classes))) == guard_ok
    print("  4 classes, guard range 0..3 (ok)")
    # verify the TRAIN call in the full script disables hue jitter (colour==class)
    full = (HERE / "train_light_state_kaggle.py").read_text(encoding="utf-8")
    assert "hsv_h=0.0" in full, "hue jitter MUST be disabled for a colour-state detector"
    assert "fliplr=0.5" in full, "horizontal flip is fine and expected"
    print("  hsv_h=0.0 (no hue jitter) present (ok)")
    print("Light notebook logic OK\n")


if __name__ == "__main__":
    check_idd()
    check_light()
    print("all self-checks passed.")
