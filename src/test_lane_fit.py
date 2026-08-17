"""Self-check for LaneFitter.

The previous version of this file generated synthetic lanes using the same
perspective assumption the fitter used, so it passed 8/8 while the fitter was
badly broken on real video. Lesson: don't test a component against your own
assumptions.

These checks instead assert the properties that were actually violated on real
footage, and the last one runs against real YOLOP masks from the demo clip.

Run: python test_lane_fit.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lane_fit import LaneFitter

W, H = 1280, 720
VIDEO = Path(r"D:\carLane\downloads\frankfurt_720p_5min.mp4")


def line_mask(x_bottom, x_top, y_lo, y_hi, w=W, h=H, thick=5):
    m = np.zeros((h, w), dtype=np.uint8)
    n = max(2, y_hi - y_lo)
    for i in range(n):
        y = y_lo + i
        x = int(x_top + (x_bottom - x_top) * i / n)
        cv2.circle(m, (x, y), thick, 1, -1)
    return m


# --- The bugs that real footage exposed ---------------------------------------

def test_no_lanes_means_no_fit():
    f = LaneFitter(W, H)
    assert not f.update(np.zeros((H, W), dtype=np.uint8))
    assert f.offset_ratio() is None


def test_does_not_redraw_a_stale_fit():
    """The original bug: one good frame then empty frames kept drawing old lines."""
    f = LaneFitter(W, H)
    good = line_mask(300, 600, 400, 715) | line_mask(1000, 700, 400, 715)
    assert f.update(good)
    assert f.offset_ratio() is not None

    assert not f.update(np.zeros((H, W), dtype=np.uint8)), "empty frame must not fit"
    assert f.offset_ratio() is None, "offset must clear when lanes vanish"
    img = np.zeros((H, W, 3), dtype=np.uint8)
    assert not f.draw(img).any(), "must draw nothing without current pixels"


def test_single_line_gives_no_offset():
    """One visible line cannot define a lane centre — must not invent one."""
    f = LaneFitter(W, H)
    f.update(line_mask(300, 600, 400, 715))
    assert f.offset_ratio() is None, "one line is not an ego pair"


def test_never_draws_outside_pixel_extent():
    """The extrapolation bug: curves were evaluated over the full frame height."""
    m = line_mask(400, 500, 500, 700)
    f = LaneFitter(W, H)
    f.update(m)
    assert f.lines, "should fit the line"

    # Compare against the extent the fitter actually sees: the brush radius widens
    # the mask, and the dash-joining close can legitimately extend it further.
    kx, ky = f.CLOSE_KERNEL
    closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)))
    ys = np.nonzero(closed)[0]
    y_lo, y_hi = int(ys.min()), int(ys.max())

    for ln in f.lines:
        pts = ln.points()
        assert pts[:, 1].min() >= y_lo - 2, f"drew above data: {pts[:,1].min()} < {y_lo}"
        assert pts[:, 1].max() <= y_hi + 2, f"drew below data: {pts[:,1].max()} > {y_hi}"
        # The whole point: the drawn span must track the data, not the frame.
        span = pts[:, 1].max() - pts[:, 1].min()
        assert span <= (y_hi - y_lo) + 4, "drew beyond the component's own extent"
        assert span < H * 0.55, f"span {span:.0f}px is most of the frame — extrapolating"


def test_separate_lines_stay_separate():
    """Midpoint splitting used to merge distinct lines into one curve."""
    f = LaneFitter(W, H)
    # Three lines all left of centre — the old code fit them as a single curve
    m = (line_mask(150, 350, 400, 715)
         | line_mask(400, 520, 400, 715)
         | line_mask(600, 640, 400, 715))
    f.update(m)
    assert len(f.lines) >= 3, f"expected >=3 distinct lines, got {len(f.lines)}"


def test_centred_reads_near_zero():
    f = LaneFitter(W, H)
    f.update(line_mask(340, 590, 400, 715) | line_mask(940, 690, 400, 715))
    off = f.offset_ratio()
    assert off is not None, "symmetric pair should yield an offset"
    assert abs(off) < 0.3, f"symmetric lanes should read ~0, got {off:.2f}"


def test_offset_sign_follows_drift():
    """Both lines shifted left => car sits right of centre => positive offset."""
    a = LaneFitter(W, H)
    a.update(line_mask(340, 590, 400, 715) | line_mask(940, 690, 400, 715))
    centred = a.offset_ratio()

    b = LaneFitter(W, H)
    b.update(line_mask(140, 390, 400, 715) | line_mask(740, 490, 400, 715))
    drifted = b.offset_ratio()

    assert drifted is not None and centred is not None
    assert drifted > centred, f"offset should rise: {centred:.2f} -> {drifted:.2f}"


def test_noise_speck_is_ignored():
    f = LaneFitter(W, H)
    m = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(m, (600, 300), 3, 1, -1)     # tiny blob, no vertical extent
    assert not f.update(m), "a speck must not count as a lane line"


def test_ego_pair_ignores_lines_high_in_frame():
    """A line that stops near the horizon can't bound the ego lane."""
    f = LaneFitter(W, H)
    m = (line_mask(300, 600, 400, 715)      # reaches the bottom
         | line_mask(1000, 700, 400, 715)   # reaches the bottom
         | line_mask(200, 210, 300, 420))   # far ahead only
    f.update(m)
    for ln in (f.left, f.right):
        if ln is not None:
            assert ln.y_hi >= H * 0.80, "ego line must reach low in the frame"


# --- Against real model output ------------------------------------------------

def test_on_real_yolop_masks():
    """Regression guard on real footage: no ego pair without two real lines,
    and every drawn point must sit inside real pixel extent."""
    if not VIDEO.exists():
        print("       (skipped: demo clip not present)")
        return
    try:
        import torch
        from trt_runner import TRTSeg
        from utils import MODELS_DIR
    except Exception as e:
        print(f"       (skipped: {e})")
        return

    eng = MODELS_DIR / "yolop_384.engine"
    if not eng.exists():
        print("       (skipped: yolop engine not built)")
        return

    device = torch.device("cuda")
    yolop = TRTSeg(eng, imgsz=384)
    cap = cv2.VideoCapture(str(VIDEO))
    f = LaneFitter(W, H)

    checked = fitted = 0
    for i in range(240):
        ret, frame = cap.read()
        if not ret:
            break
        if i % 20:
            continue
        img = cv2.cvtColor(cv2.resize(frame, (384, 384)), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img).to(device).permute(2, 0, 1).unsqueeze(0)
        t = t.float().div_(255.0).contiguous()
        _, ll_t = yolop.infer(t)
        ll = cv2.resize(ll_t.squeeze().cpu().numpy(), (W, H),
                        interpolation=cv2.INTER_NEAREST)

        ok = f.update(ll)
        checked += 1
        n_px = int((ll == 1).sum())

        if n_px < 45:
            assert not ok, f"frame {i}: fitted with only {n_px} lane pixels"
            assert f.offset_ratio() is None, f"frame {i}: offset without pixels"
        if ok:
            fitted += 1
            for ln in f.lines:
                pts = ln.points()
                assert pts[:, 1].min() >= ln.y_lo - 2, f"frame {i}: drew above data"
                assert pts[:, 1].max() <= ln.y_hi + 2, f"frame {i}: drew below data"
            if f.offset_ratio() is not None:
                assert f.left is not None and f.right is not None, \
                    f"frame {i}: offset reported without an ego pair"
                assert abs(f.offset_ratio()) < 3, \
                    f"frame {i}: implausible offset {f.offset_ratio():.2f}"
    cap.release()
    assert checked > 0
    print(f"       ({fitted}/{checked} real frames produced a fit)")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
