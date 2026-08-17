"""Traffic-light filtering and state classification with temporal voting.

Measured problems this fixes (over 900 Frankfurt frames):
  * 53% of accepted detections sat in the LOWER HALF of the frame. Traffic lights
    hang above the roadway, so those are taillights, reflections and signs.
  * 11% of detections changed colour between consecutive frames. A real light
    holds a colour for seconds, so that rate is classifier noise, not reality.
  * 22 detections appeared for a single frame then vanished.
"""

import numpy as np


class TrafficLightTracker:
    """Filters candidate lights on geometry, then votes their state over time.

    ponytail: association is by coarse grid cell, not IoU tracking. Ceiling — two
    lights within one cell merge into a single track, and a fast lateral pass can
    hand one light's history to another. Upgrade path = IoU/ByteTrack association.
    """

    # Geometry gates
    MIN_H = 12              # Distant lights are genuinely this small
    MAX_ASPECT = 0.95       # A 3-lamp head is clearly taller than wide
    MAX_Y_FRAC = 0.55       # Must sit above this fraction of frame height
    MIN_CONF = 0.20

    # Temporal voting
    VOTES_TO_SWITCH = 2     # Consistent readings needed to change a held colour
    TTL = 4                 # Cycles a track survives without a fresh detection
    CELL = 48               # Grid cell size for association

    def __init__(self, frame_h):
        self.h = frame_h
        self.tracks = {}    # cell -> dict(state, pending, pending_n, ttl, box, conf)

    # --- filtering --------------------------------------------------------

    def accept(self, x1, y1, x2, y2, conf):
        """Geometry gate. Returns False for things that can't be a traffic light."""
        bh, bw = y2 - y1, x2 - x1
        if bh < self.MIN_H or bw <= 0:
            return False
        if bw / bh > self.MAX_ASPECT:
            return False
        # The single biggest false-positive source: detections below the horizon.
        if y1 > self.h * self.MAX_Y_FRAC:
            return False
        if conf < self.MIN_CONF:
            return False
        return True

    # --- state from pixels ------------------------------------------------

    @staticmethod
    def classify(roi):
        """Which third of the lamp head is lit, by colour opponency."""
        if roi is None or roi.size == 0 or roi.shape[0] < 6:
            return None
        h = roi.shape[0]
        th = max(1, h // 3)
        top, mid, bot = roi[:th], roi[th:2 * th], roi[2 * th:]
        if min(top.size, mid.size, bot.size) == 0:
            return None

        # Red-vs-green opponency per cell, plus overall brightness
        top_r = float(top[:, :, 2].mean()) - float(top[:, :, 1].mean())
        bot_g = float(bot[:, :, 1].mean()) - float(bot[:, :, 2].mean())
        mid_b = float(mid[:, :, 1].mean()) + float(mid[:, :, 2].mean())
        top_b = float(top[:, :, 2].mean()) + float(top[:, :, 1].mean())
        bot_b = float(bot[:, :, 1].mean()) + float(bot[:, :, 2].mean())

        # Require the winning cell to be the brightest as well as the most
        # colour-shifted — a dark red-ish cell isn't a lit red lamp.
        if bot_g > 10 and bot_b >= top_b:
            return "GREEN"
        if top_r > 10 and top_b >= bot_b:
            return "RED"
        if mid_b > max(top_b, bot_b) + 25:
            return "YELLOW"
        return None

    # --- per-frame update -------------------------------------------------

    def update(self, detections, frame):
        """detections: iterable of (x1, y1, x2, y2, conf) in frame coords.

        Returns list of (x1, y1, x2, y2, state, conf) for display.
        """
        for tr in self.tracks.values():
            tr["ttl"] -= 1

        for (x1, y1, x2, y2, conf) in detections:
            if not self.accept(x1, y1, x2, y2, conf):
                continue
            cell = (x1 // self.CELL, y1 // self.CELL)
            reading = self.classify(frame[max(0, y1):y2, max(0, x1):x2])

            tr = self.tracks.get(cell)
            if tr is None:
                tr = {"state": reading, "pending": None, "pending_n": 0,
                      "ttl": self.TTL, "box": (x1, y1, x2, y2), "conf": conf,
                      "seen": 1}
                self.tracks[cell] = tr
                continue

            tr["box"] = (x1, y1, x2, y2)
            tr["conf"] = conf
            tr["ttl"] = self.TTL
            tr["seen"] += 1

            if reading is None:
                continue
            if tr["state"] is None:
                tr["state"] = reading
            elif reading == tr["state"]:
                tr["pending"], tr["pending_n"] = None, 0
            else:
                # Require repeated agreement before flipping a held colour
                if reading == tr["pending"]:
                    tr["pending_n"] += 1
                else:
                    tr["pending"], tr["pending_n"] = reading, 1
                if tr["pending_n"] >= self.VOTES_TO_SWITCH:
                    tr["state"] = reading
                    tr["pending"], tr["pending_n"] = None, 0

        # Drop expired, and never surface a track seen only once (ghost)
        self.tracks = {k: v for k, v in self.tracks.items() if v["ttl"] > 0}

        out = []
        for tr in self.tracks.values():
            if tr["seen"] < 2 or tr["state"] is None:
                continue
            x1, y1, x2, y2 = tr["box"]
            out.append((x1, y1, x2, y2, tr["state"], tr["conf"]))
        return out

    def dominant_state(self, lights):
        """State of the largest visible light — the one most likely to apply to us."""
        if not lights:
            return ""
        biggest = max(lights, key=lambda b: (b[3] - b[1]) * (b[2] - b[0]))
        return biggest[4]
