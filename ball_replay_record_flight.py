#!/usr/bin/env python3
"""Record USB camera during an unchanged SO-101 replay."""

import builtins
import csv
import importlib.util
import sys
import threading
import time
from pathlib import Path

import cv2
import ball_replay_camera_ready as bridge


DURATION = 8.0
VIDEO_FILE = "flight_camera.avi"
CSV_FILE = "flight_timestamps.csv"
INFO_FILE = "flight_camera_info.txt"


class FlightRecorder:
    def __init__(self):
        self.thread = None
        self.stop_event = threading.Event()
        self.start_t = None
        self.motion_t = None
        self.frames = []
        self.error = None
        self.frame_count = 0

    def start(self):
        self.cap = cv2.VideoCapture(
            "/dev/video0", cv2.CAP_V4L2
        )
        if not self.cap.isOpened():
            raise RuntimeError("Camera open failed")

        self.cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG")
        )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 25)

        # Warm up camera.
        frame = None
        for _ in range(5):
            ok, img = self.cap.read()
            if ok:
                frame = img

        if frame is None:
            self.cap.release()
            raise RuntimeError("Cannot capture camera frame")

        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")

        self.writer = cv2.VideoWriter(
            VIDEO_FILE, fourcc, 25.0, (w, h)
        )
        if not self.writer.isOpened():
            self.cap.release()
            raise RuntimeError("VideoWriter open failed")

        self.start_t = time.monotonic()

        self.thread = threading.Thread(
            target=self._capture,
            daemon=False
        )
        self.thread.start()

        print(
            f"[CAM] Recording started: "
            f"{w}x{h}, max {DURATION:.0f}s"
        )

    def _capture(self):
        try:
            while not self.stop_event.is_set():
                if time.monotonic() - self.start_t >= DURATION:
                    break

                ok, frame = self.cap.read()
                t = time.monotonic()

                if not ok or frame is None:
                    continue

                self.writer.write(frame)
                self.frames.append(
                    (self.frame_count, t - self.start_t)
                )
                self.frame_count += 1

        except Exception as e:
            self.error = repr(e)

        finally:
            self.writer.release()
            self.cap.release()

    def stop(self):
        if self.thread is None:
            return

        self.stop_event.set()
        self.thread.join()

        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["frame", "camera_time_s", "from_replay_start_s"]
            )

            offset = (
                self.motion_t - self.start_t
                if self.motion_t is not None else 0.0
            )

            for i, t in self.frames:
                writer.writerow(
                    [i, round(t, 6), round(t - offset, 6)]
                )

        elapsed = (
            self.frames[-1][1] - self.frames[0][1]
            if len(self.frames) > 1 else 0
        )
        fps = (
            (len(self.frames) - 1) / elapsed
            if elapsed > 0 else 0
        )

        report = (
            f"Recorded frames: {self.frame_count}\n"
            f"Duration: {elapsed:.3f} s\n"
            f"Measured FPS: {fps:.2f}\n"
            f"Replay marker offset: {offset:.3f} s\n"
            f"Camera error: {self.error}\n"
        )

        Path(INFO_FILE).write_text(report)
        print("\n=== FLIGHT CAMERA RECORDING ===")
        print(report)
        print("Video:", VIDEO_FILE)
        print("CSV:", CSV_FILE)

        self.thread = None


def main():
    original_input = builtins.input
    recorder = FlightRecorder()

    script = Path(
        "ball_record_replay_id5off_v3.py"
    ).resolve()

    spec = importlib.util.spec_from_file_location(
        "so101_original_replay", script
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def checked_input(prompt=""):
        if "Press Enter to start REPLAY" in prompt:
            bridge.wait_for_ready()

            original_input(
                "\nREADY confirmed. "
                "Press Enter to START REPLAY "
                "(Ctrl+C to abort)... "
            )

            # Recording begins immediately before replay.
            recorder.start()
            recorder.motion_t = time.monotonic()
            print("[CAM] Replay start marker")
            return ""

        return original_input(prompt)

    builtins.input = checked_input

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
        recorder.stop()


if __name__ == "__main__":
    main()
