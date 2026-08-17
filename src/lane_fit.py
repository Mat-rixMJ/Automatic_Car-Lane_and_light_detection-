"""Turn YOLOP's lane mask into clean per-line curves.

Design notes (learned the hard way — see memory.md):

An earlier version warped the mask to bird's-eye, split pixels at the image
midpoint and fit one polynomial per side. That failed on real footage because:
  * YOLOP marks EVERY lane line in view, not just the ego lane's two, so a
    midpoint split fits a single curve through pixels from several lines.
  * Lane pixels only occupy roughly the bottom third of the frame, but the curve
    was evaluated over the full height, so most of each line was extrapolated.
  * A side with zero pixels kept drawing its previous fit, producing lines that
    wandered independently of the road.

This version avoids all three: it segments the mask into connected components
(one per painted line), fits each component only across its own vertical extent,
and draws nothing that lacks current pixel support.
"""

import cv2
import numpy as np


class LaneLine:
    """One fitted lane line, valid only across the y-range it was fitted on."""

    __slots__ = ("coef", "y_lo", "y_hi", "n_px")

    def __init__(self, coef, y_lo, y_hi, n_px):
        self.coef = coef
        self.y_lo = y_lo
        self.y_hi = y_hi
        self.n_px = n_px

    def x_at(self, y):
        return float(np.polyval(self.coef, y))

    def points(self, step=12):
        ys = np.arange(self.y_lo, self.y_hi + 1, step, dtype=np.float32)
        xs = np.polyval(self.coef, ys)
        return np.stack([xs, ys], axis=1)


class LaneFitter:
    """Fits every visible lane line, then identifies the ego lane pair.

    ponytail: fits in image space with a 2nd-order poly per connected component.
    Ceiling — a single painted line broken into dashes becomes several components,
    so long dashed lines are drawn as separate segments rather than one line.
    Upgrade path = merge components that share a fitted trajectory before fitting.
    """

    # Tuned by sweep over 400 real YOLOP masks (src/tune_lane_fit.py). The
    # residual gate dominated: 12 -> 20 raised the ego-pair rate 22% -> 32%,
    # while taller close kernels alone gained almost nothing. Kernel kept at 41px
    # rather than 61-81px, which only added ~2% but risks bridging separate lines.
    MIN_PX = 40              # Components smaller than this are noise
    MIN_SPAN = 15            # Need real vertical extent to fit a direction
    MAX_RESID = 20           # Mean |fit - pixel| px before the component is junk
    CLOSE_KERNEL = (5, 41)   # Tall/narrow: joins dashes, won't bridge lanes
    SMOOTH = 0.6             # EMA on the ego-lane offset only

    def __init__(self, frame_w, frame_h):
        self.w = frame_w
        self.h = frame_h
        self.lines = []
        self.left = None
        self.right = None
        self._offset = None

    def update(self, lane_mask, kernel=None, max_resid=None):
        """lane_mask: uint8 HxW, 1 where lane. Returns True if any line was fitted.

        Nothing is retained from previous frames — a frame with no lane pixels
        reports no lines rather than redrawing an old fit.
        """
        self.lines = []
        self.left = None
        self.right = None
        kx, ky = kernel or self.CLOSE_KERNEL
        max_resid = self.MAX_RESID if max_resid is None else max_resid

        # Close gaps ALONG the line direction so dashed markings join into one
        # component. Tall-and-narrow so it doesn't bridge adjacent lanes.
        mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)))

        n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for lab in range(1, n_lab):
            if stats[lab, cv2.CC_STAT_AREA] < self.MIN_PX:
                continue
            if stats[lab, cv2.CC_STAT_HEIGHT] < self.MIN_SPAN:
                continue

            ys, xs = np.nonzero(labels == lab)
            y_lo, y_hi = int(ys.min()), int(ys.max())

            # Fit x = f(y): lane lines are near-vertical in image space, so y is
            # the well-conditioned independent variable.
            order = 2 if (y_hi - y_lo) > 60 else 1
            try:
                coef = np.polyfit(ys, xs, order)
            except (np.linalg.LinAlgError, ValueError):
                continue
            if order == 1:
                coef = np.array([0.0, coef[0], coef[1]])

            # Reject fits that don't track their own pixels
            if np.abs(np.polyval(coef, ys) - xs).mean() > max_resid:
                continue

            self.lines.append(LaneLine(coef, y_lo, y_hi, len(xs)))

        self._identify_ego()
        return bool(self.lines)

    def _identify_ego(self):
        """Pick the nearest line either side of the camera axis, low in the frame."""
        centre = self.w / 2
        probe_y = self.h * 0.93        # Just ahead of the bumper

        best_l = best_r = None
        dl = dr = 1e9
        for ln in self.lines:
            # Only lines that reach near the bottom can bound the ego lane.
            # Measured: lowering from 0.80 to 0.70 raised ego-pair rate
            # Frankfurt 57%->72%, BDDA highway 20%->43%.
            if ln.y_hi < self.h * 0.70:
                continue
            x = ln.x_at(min(probe_y, ln.y_hi))
            if not (-self.w < x < 2 * self.w):
                continue
            d = centre - x
            if 0 < d < dl:
                dl, best_l = d, ln
            elif -dr < d <= 0 and -d < dr:
                dr, best_r = -d, ln

        self.left, self.right = best_l, best_r

        if best_l is None or best_r is None:
            self._offset = None
            return
        lx = best_l.x_at(min(probe_y, best_l.y_hi))
        rx = best_r.x_at(min(probe_y, best_r.y_hi))
        width = rx - lx
        if width < self.w * 0.15:      # Implausibly narrow -> not an ego lane pair
            self._offset = None
            return
        raw = (centre - (lx + rx) / 2) / (width / 2)
        # Smooth the scalar, not the geometry, so lines stay tied to real pixels
        self._offset = raw if self._offset is None else \
            self.SMOOTH * self._offset + (1 - self.SMOOTH) * raw

    def draw(self, img, fill=True):
        """Draw fitted lines. Ego-lane pair highlighted, others dimmer."""
        if not self.lines:
            return img

        if fill and self.left is not None and self.right is not None:
            lp = self.left.points()
            rp = self.right.points()
            y0 = max(self.left.y_lo, self.right.y_lo)
            lp = lp[lp[:, 1] >= y0]
            rp = rp[rp[:, 1] >= y0]
            if len(lp) > 1 and len(rp) > 1:
                poly = np.vstack([lp, rp[::-1]]).astype(np.int32)
                layer = np.zeros_like(img)
                cv2.fillPoly(layer, [poly], (0, 70, 0))
                cv2.add(img, layer, dst=img)

        for ln in self.lines:
            is_ego = ln is self.left or ln is self.right
            colour = (0, 255, 255) if is_ego else (140, 200, 200)
            thick = 6 if is_ego else 2
            pts = ln.points().astype(np.int32)
            if len(pts) > 1:
                cv2.polylines(img, [pts], False, colour, thick, cv2.LINE_AA)
        return img

    def offset_ratio(self):
        """Ego offset from lane centre: 0 = centred, +/-1 = on a line.

        None when the ego lane pair isn't currently visible.
        """
        return self._offset

    def stats(self):
        return {
            "lines": len(self.lines),
            "has_ego_pair": self.left is not None and self.right is not None,
            "offset": self._offset,
        }

    def use_da_fallback(self, da_mask):
        """Use drivable-area edges as ego-lane boundaries when lane lines fail.

        On highways the road edge IS the lane boundary — there's a painted line
        on one side and the road edge on the other. YOLOP's DA mask gives us that
        edge reliably (measured: 79% on BDDA highway vs 20% from lines alone).

        Only activates when self.left or self.right is None after update().
        """
        if self.left is not None and self.right is not None:
            return  # Already have a pair from lane lines

        # Scan the DA mask from y=50% to y=95% of frame for left/right edges
        y_start = int(self.h * 0.50)
        y_end = int(self.h * 0.95)
        step = 6
        left_pts = []
        right_pts = []

        for y in range(y_start, y_end, step):
            row = da_mask[y]
            nz = np.nonzero(row)[0]
            if len(nz) < 20:
                continue
            left_pts.append((float(nz[0]), float(y)))
            right_pts.append((float(nz[-1]), float(y)))

        if len(left_pts) < 5 or len(right_pts) < 5:
            return

        # Verify the width is plausible for a single lane (not whole multi-lane road)
        l_bot = np.mean([p[0] for p in left_pts[-3:]])
        r_bot = np.mean([p[0] for p in right_pts[-3:]])
        width = r_bot - l_bot

        # ponytail: On multi-lane roads the DA covers ALL lanes, so width > 0.6*frame
        # means this isn't a single-lane corridor. Narrow it by moving edges inward
        # toward the camera axis. Ceiling: assumes ego is near frame centre.
        centre = self.w / 2
        if width > self.w * 0.55:
            # Too wide — shrink toward the two nearest edges to frame centre
            # Use only points within 40% of centre on each side
            left_pts = [(x, y) for x, y in left_pts if x > centre - self.w * 0.45]
            right_pts = [(x, y) for x, y in right_pts if x < centre + self.w * 0.45]
            if len(left_pts) < 5 or len(right_pts) < 5:
                return

        # Create synthetic LaneLine objects from the DA edges
        if self.left is None and left_pts:
            ys = np.array([p[1] for p in left_pts])
            xs = np.array([p[0] for p in left_pts])
            try:
                coef = np.polyfit(ys, xs, 2)
                self.left = LaneLine(coef, int(ys.min()), int(ys.max()), len(xs))
            except (np.linalg.LinAlgError, ValueError):
                pass

        if self.right is None and right_pts:
            ys = np.array([p[1] for p in right_pts])
            xs = np.array([p[0] for p in right_pts])
            try:
                coef = np.polyfit(ys, xs, 2)
                self.right = LaneLine(coef, int(ys.min()), int(ys.max()), len(xs))
            except (np.linalg.LinAlgError, ValueError):
                pass

        # Recompute offset if we now have a pair
        if self.left is not None and self.right is not None:
            probe_y = self.h * 0.93
            lx = self.left.x_at(min(probe_y, self.left.y_hi))
            rx = self.right.x_at(min(probe_y, self.right.y_hi))
            width = rx - lx
            if width > self.w * 0.15:
                raw = (centre - (lx + rx) / 2) / (width / 2)
                self._offset = raw if self._offset is None else \
                    self.SMOOTH * self._offset + (1 - self.SMOOTH) * raw
