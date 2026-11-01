#!/usr/bin/env python3
"""SO-101: camera READY check before original replay."""

import builtins
import importlib.util
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# ---- Camera settings ----
DEVICE = "/dev/video0"
WIDTH = 640
HEIGHT = 480
FPS = 25

# Validated ball detection settings
THRESHOLD = 65
import json
from pathlib import Path

_CALIB = json.loads(
    Path(__file__).with_name(
        "ball_camera_calibration.json"
    ).read_text(encoding="utf-8")
)
ROI = tuple(_CALIB["roi"])
REFERENCE_Y = float(_CALIB["reference_y"])

# Initial position measured previously
REFERENCE_X = float(_CALIB["reference_x"])

# READY criteria
SIGMA_LIMIT = 4.0
CENTER_LIMIT = 10.0
STABLE_SECONDS = 1.0
WINDOW_SECONDS = 1.0

MIN_SAMPLES = 15
MAX_MONITOR_SECONDS = 45.0


def detect_ball(frame):
    x1, y1, x2, y2 = ROI

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    crop = gray[y1:y2, x1:x2]

    _, mask = cv2.threshold(
        crop, THRESHOLD, 255,
        cv2.THRESH_BINARY_INV
    )

    n, labels, stats, centers = \
        cv2.connectedComponentsWithStats(mask)

    candidates = []

    for i in range(1, n):
        x, y, w, h, area = stats[i]

        if not (120 <= area <= 1500):
            continue

        if not (14 <= w <= 55 and 14 <= h <= 55):
            continue

        if not (0.7 <= w / h <= 1.3):
            continue

        fill = area / (w * h)

        if not (0.55 <= fill <= 0.90):
            continue

        cx = float(centers[i][0] + x1)
        cy = float(centers[i][1] + y1)

        score = abs(cx - REFERENCE_X) + abs(cy - REFERENCE_Y)
        candidates.append((score, cx, cy))

    if not candidates:
        return None

    _, cx, cy = min(candidates)
    return cx, cy


def wait_for_ready():
    print("\n=== CAMERA READY CHECK ===")
    print("Reference X:", REFERENCE_X)
    print("Sigma limit:", SIGMA_LIMIT, "px")
    print("Center limit:", CENTER_LIMIT, "px")
    print("Monitoring ball movement...")

    cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    try:
        cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG")
        )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)

        history = deque()
        stable_since = None
        start = time.monotonic()
        last_report = start
        last_valid = None

        while True:
            ok, frame = cap.read()
            now = time.monotonic()

            if now - start > MAX_MONITOR_SECONDS:
                raise RuntimeError(
                    "READY timeout. Replay not started."
                )

            if not ok or frame is None:
                history.clear()
                stable_since = None
                time.sleep(0.04)
                continue

            pos = detect_ball(frame)

            if pos is None:
                history.clear()
                stable_since = None

                if now - last_report >= 1.0:
                    print(
                        f"t={now-start:5.1f}s "
                        "BALL NOT DETECTED",
                        flush=True
                    )
                    last_report = now

                continue

            x, y = pos
            last_valid = (x, y)
            history.append((now, x))

            while history and \
                  now - history[0][0] > WINDOW_SECONDS:
                history.popleft()

            if len(history) < MIN_SAMPLES:
                continue

            # Require approximately one full second of samples.
            if now - history[0][0] < 0.85:
                continue

            xs = np.array(
                [item[1] for item in history],
                dtype=float
            )

            sigma = float(np.std(xs))
            mean_x = float(np.mean(xs))

            stable = (
                sigma <= SIGMA_LIMIT
                and abs(mean_x - REFERENCE_X) <= CENTER_LIMIT
                and abs(x - REFERENCE_X) <= CENTER_LIMIT
            )

            if stable:
                if stable_since is None:
                    stable_since = now

                if now - stable_since >= STABLE_SECONDS:
                    print(
                        f"\n*** READY *** "
                        f"X={x:.2f}, "
                        f"mean={mean_x:.2f}, "
                        f"sigma={sigma:.2f}px"
                    )
                    return
            else:
                stable_since = None

            if now - last_report >= 0.5:
                state = (
                    "STABILIZING" if stable else "WAIT"
                )
                print(
                    f"t={now-start:5.1f}s "
                    f"X={x:6.1f} "
                    f"mean={mean_x:6.1f} "
                    f"sigma={sigma:5.2f} "
                    f"{state}"
                )
                last_report = now

    finally:
        cap.release()
        print("Camera released.")


def main():
    original_input = builtins.input

    script = Path(
        "ball_record_replay_id5off_v3.py"
    ).resolve()

    if not script.exists():
        raise FileNotFoundError(script)

    spec = importlib.util.spec_from_file_location(
        "so101_original_replay", script
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def camera_checked_input(prompt=""):
        if "Press Enter to start REPLAY" in prompt:
            wait_for_ready()

            return original_input(
                "\nREADY confirmed. "
                "Press Enter to START REPLAY "
                "(Ctrl+C to abort)... "
            )

        return original_input(prompt)

    # Original replay runs in this same process.
    # Only the replay-start confirmation is intercepted.
    builtins.input = camera_checked_input

    try:
        sys.argv = [
            str(script),
            "replay",
            "--file",
            "ball_throw_D_wrist25_minus.json",
            "--auto-position",
            "--speed",
            "1.0",
        ]
        module.main()

    finally:
        builtins.input = original_input


if __name__ == "__main__":
    main()
