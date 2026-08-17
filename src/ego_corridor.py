"""Build a stable ego-lane corridor from YOLOP's drivable-area mask.

Why not lane lines: measured over ~2800 frames of Frankfurt + BDDA footage, only
4 frames produced two fitted lane lines that both span a common height range.
YOLOP marks lane pixels well, but on real footage the ego lane's two boundaries
are almost never both visible and unbroken at the same time (occlusion, dashes,
single-sided markings, intersections). So a corridor built by pairing lane lines
is not achievable on this data.

The drivable-area mask, by contrast, is present in >92% of frames. This class
takes the road surface and carves the ego lane out of it:

  per scanline: road span from DA mask
  -> centre-line, tracked from the bottom of the frame upward
  -> half-width clamped by a perspective prior so it stays ONE lane
  -> temporal EMA on control points so the corridor stops jittering

ponytail: model-based corridor refined by the DA mask, not a true two-boundary
detection. Ceiling — it assumes the ego lane is the part of the road surface
directly ahead of the camera, so it cannot represent a lane change in progress
and will lag briefly when the car crosses into a new lane. Upgrade path = a lane
model trained to output ego-lane boundaries directly (e.g. CLRNet/UFLD).
"""

import cv2
import numpy as np


class EgoCorridor:
    # Scan band: below the horizon, above the bonnet
    Y_TOP = 0.60
    Y_BOT = 0.97
    N_ROWS = 22

    # Perspective prior for ONE lane, as a fraction of frame width.
    # Measured from genuine line-derived pairs: ~44% at y=95%h. A lane narrows
    # toward the vanishing point, so the prior scales with depth into the frame.
    W_AT_BOTTOM = 0.46
    W_AT_TOP = 0.12

    MIN_ROAD_PX = 40        # Scanline needs this much road to be usable
    SMOOTH = 0.80           # EMA on control points — high, because jitter was the
                            # dominant visual defect (34-62px/frame before)
    MAX_MISSES = 8          # Frames without road before the corridor expires

    # Vehicles occlude the road, splitting the DA mask into fragments. Without
    # bridging, per-row "longest run" lands on a sliver beside a car and the
    # corridor collapses (measured: 7% of frame width on BDDA1003).
    OCCL_BRIDGE = 121       # Horizontal close width, in px, to span vehicles
    GAP_TOL = 18            # Pixels of gap tolerated inside one run
    # Mild floor only. Bridging above already prevents the sliver case, and a
    # hard floor forces the corridor wider than the road, which showed up as
    # 21-31% green spill onto pavement.
    MIN_W_FRAC = 0.30

    def __init__(self, frame_w, frame_h):
        self.w = frame_w
        self.h = frame_h
        self.ys = np.linspace(frame_h * self.Y_TOP, frame_h * self.Y_BOT, self.N_ROWS)
        self._left = None       # smoothed x per scanline
        self._right = None
        self._misses = 0
        self._offset = None
        # Bottom anchor carried across frames. Re-anchoring to frame centre every
        # frame made the corridor snap to a run edge whenever the road run didn't
        # contain the centre (measured 52px/frame jitter on BDDA1003).
        self._anchor = float(frame_w) / 2

    # --- geometry helpers -------------------------------------------------

    def _prior_halfwidth(self, y):
        """Half-width of one lane at image row y, from the perspective prior."""
        t = (y / self.h - self.Y_TOP) / (self.Y_BOT - self.Y_TOP)
        t = float(np.clip(t, 0.0, 1.0))
        frac = self.W_AT_TOP + (self.W_AT_BOTTOM - self.W_AT_TOP) * t
        return self.w * frac / 2.0

    def _road_span(self, da, y, anchor):
        """Road run on row y that contains `anchor`, as (x_start, x_end).

        Picks the run the vehicle is actually in rather than the longest one —
        the longest run is frequently a fragment beside an occluding vehicle.
        """
        row = da[int(y)]
        nz = np.nonzero(row)[0]
        if len(nz) < self.MIN_ROAD_PX:
            return None
        breaks = np.nonzero(np.diff(nz) > self.GAP_TOL)[0]
        runs = [r for r in np.split(nz, breaks + 1) if len(r) >= self.MIN_ROAD_PX]
        if not runs:
            return None
        # Prefer the run containing the anchor; else the run nearest to it.
        for r in runs:
            if r[0] <= anchor <= r[-1]:
                return float(r[0]), float(r[-1])
        best = min(runs, key=lambda r: min(abs(r[0] - anchor), abs(r[-1] - anchor)))
        return float(best[0]), float(best[-1])

    # --- main update ------------------------------------------------------

    def update(self, da_mask, lane_mask=None):
        """Rebuild the corridor. Returns True when a corridor is available."""
        # Bridge vehicle-shaped holes in the road mask before scanning
        road = cv2.morphologyEx(
            da_mask, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (self.OCCL_BRIDGE, 9)))

        rows_l = np.full(self.N_ROWS, np.nan)
        rows_r = np.full(self.N_ROWS, np.nan)

        # Walk bottom-up so the centre is anchored near the car, where we know
        # the ego lane is directly ahead, then track the road as it curves away.
        prev_centre = None
        for i in range(self.N_ROWS - 1, -1, -1):
            y = self.ys[i]
            anchor = self._anchor if prev_centre is None else prev_centre
            span = self._road_span(road, y, anchor)
            if span is None:
                continue
            x0, x1 = span
            half = self._prior_halfwidth(y)

            if prev_centre is None:
                # Start from where the corridor was last frame, pulled gently
                # back toward the camera axis, so it tracks instead of snapping.
                target = 0.75 * self._anchor + 0.25 * (self.w / 2)
                centre = float(np.clip(target, x0, x1))
            else:
                # Follow the road, but keep the corridor continuous up the frame.
                centre = float(np.clip(prev_centre, x0, x1))

            l = max(x0, centre - half)
            r = min(x1, centre + half)
            # Width floor: a genuinely narrow road run must not shrink the
            # corridor to a sliver. Re-centre and take the prior width instead.
            if (r - l) < 2 * half * self.MIN_W_FRAC:
                want = half * self.MIN_W_FRAC
                l, r = centre - want, centre + want
            rows_l[i] = l
            rows_r[i] = r
            prev_centre = (l + r) / 2

        valid = ~np.isnan(rows_l)
        if valid.sum() < 6:
            self._misses += 1
            if self._misses > self.MAX_MISSES:
                self._left = self._right = None
                self._offset = None
            return self._left is not None
        self._misses = 0

        # Fill gaps by interpolation so the polygon has no holes
        idx = np.arange(self.N_ROWS)
        rows_l = np.interp(idx, idx[valid], rows_l[valid])
        rows_r = np.interp(idx, idx[valid], rows_r[valid])

        # Snap to a nearby lane line when one exists — this is where real lane
        # markings improve the corridor, without being required for it to exist.
        if lane_mask is not None:
            rows_l, rows_r = self._snap_to_lines(lane_mask, rows_l, rows_r)

        # Temporal EMA: the single biggest visual defect was jitter, so smooth
        # hard. Reset instantly if the corridor moves impossibly far (lane change).
        if self._left is None:
            self._left, self._right = rows_l, rows_r
        else:
            jump = abs(np.median(rows_l - self._left))
            a = 0.35 if jump > self.w * 0.20 else self.SMOOTH
            self._left = a * self._left + (1 - a) * rows_l
            self._right = a * self._right + (1 - a) * rows_r

        self._anchor = float((self._left[-1] + self._right[-1]) / 2)
        self._update_offset()
        return True

    def _snap_to_lines(self, lane_mask, rows_l, rows_r, tol_frac=0.06):
        """Pull a corridor edge onto a real lane marking if one is close."""
        tol = self.w * tol_frac
        for i, y in enumerate(self.ys):
            row = lane_mask[int(y)]
            nz = np.nonzero(row)[0]
            if len(nz) == 0:
                continue
            # left edge
            d = np.abs(nz - rows_l[i])
            if d.min() < tol:
                rows_l[i] = 0.5 * rows_l[i] + 0.5 * float(nz[d.argmin()])
            # right edge
            d = np.abs(nz - rows_r[i])
            if d.min() < tol:
                rows_r[i] = 0.5 * rows_r[i] + 0.5 * float(nz[d.argmin()])
        return rows_l, rows_r

    def _update_offset(self):
        if self._left is None:
            self._offset = None
            return
        l, r = self._left[-1], self._right[-1]
        width = r - l
        if width < self.w * 0.05:
            self._offset = None
            return
        self._offset = float((self.w / 2 - (l + r) / 2) / (width / 2))

    # --- output -----------------------------------------------------------

    def polygon(self):
        if self._left is None:
            return None
        left = np.stack([self._left, self.ys], axis=1)
        right = np.stack([self._right, self.ys], axis=1)
        return np.vstack([left, right[::-1]]).astype(np.int32)

    def draw(self, img, fill=True):
        poly = self.polygon()
        if poly is None:
            return img
        if fill:
            layer = np.zeros_like(img)
            cv2.fillPoly(layer, [poly], (0, 80, 0))
            cv2.add(img, layer, dst=img)
        n = self.N_ROWS
        left = poly[:n]
        right = poly[n:][::-1]
        cv2.polylines(img, [left], False, (0, 255, 255), 5, cv2.LINE_AA)
        cv2.polylines(img, [right], False, (0, 255, 255), 5, cv2.LINE_AA)
        return img

    def offset_ratio(self):
        return self._offset

    def width_at_bottom(self):
        if self._left is None:
            return None
        return float(self._right[-1] - self._left[-1])
