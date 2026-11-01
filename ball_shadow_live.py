#!/usr/bin/env python3
"""Online ball prediction during unchanged SO-101 replay. NO motor corrections.
Uses ball_replay_record_flight.py for original servo/control safety path.
"""
import csv
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
import ball_replay_record_flight as original

MODEL_PATH = Path('ball_ai_result/ball_predictor.pt')
OUT_CSV = Path('shadow_predictions.csv')


def detector(frame, previous, lost, elapsed):
    h_img, w_img = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, 0, 85)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    choices = []
    for k in range(1, count):
        x, y, w, h, area = map(int, stats[k])
        if not (90 <= area <= 1500 and 10 <= w <= 50 and 10 <= h <= 50):
            continue
        if not (0.48 <= w / h <= 1.65):
            continue
        if x < 275 or x + w > w_img - 7 or y < 7 or y + h > h_img - 7:
            continue
        pad = 6
        xa, xb = max(0, x-pad), min(w_img, x+w+pad)
        ya, yb = max(0, y-pad), min(h_img, y+h+pad)
        patch = gray[ya:yb, xa:xb]
        ring = np.ones(patch.shape, dtype=bool)
        ring[y-ya:y-ya+h, x-xa:x-xa+w] = False
        if not ring.any() or (patch[ring] > 105).mean() < 0.55:
            continue
        cx, cy = map(float, centers[k])
        choices.append((cx, cy))
    if previous is not None:
        reach = 64 if lost < 2 else 85
        near = [p for p in choices if np.hypot(p[0]-previous[0], p[1]-previous[1]) < reach]
        return min(near, key=lambda p: np.hypot(p[0]-previous[0], p[1]-previous[1])) if near else None
    if elapsed < 1.4:
        ref = (323.4, 396.6)
        near = [p for p in choices if np.hypot(p[0]-ref[0], p[1]-ref[1]) < 65]
        return min(near, key=lambda p: np.hypot(p[0]-ref[0], p[1]-ref[1])) if near else None
    return choices[0] if len(choices) == 1 else None


class ShadowRecorder(original.FlightRecorder):
    def __init__(self):
        super().__init__()
        self.model = None
        self.shadow_rows = []
        self.pending = {}  # future_frame -> row index
        self.positions = deque(maxlen=5)
        self.detected = 0
        self.predicted = 0
        self.prediction_ms = []

    def load_model(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f'学習済みモデルがありません: {MODEL_PATH}')
        torch.set_num_threads(1)
        checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        window = int(checkpoint['window'])
        if window != 5 or int(checkpoint['horizon']) != 3:
            raise ValueError('この実験コードはwindow=5,horizon=3に対応しています')
        self.net = nn.Sequential(
            nn.Linear(11, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 2)
        )
        self.net.load_state_dict(checkpoint['state_dict'])
        self.net.eval()
        self.x_mean = np.asarray(checkpoint['x_mean'], np.float32)
        self.x_std = np.asarray(checkpoint['x_std'], np.float32)
        self.y_mean = np.asarray(checkpoint['y_mean'], np.float32)
        self.y_std = np.asarray(checkpoint['y_std'], np.float32)
        self.train_start = float(checkpoint['start'])
        print(f'[AI] モデル読込完了: {MODEL_PATH} (CPU推論・シャドーモード)')

    def start(self):
        self.load_model()
        super().start()

    def _capture(self):
        prev = None
        lost = 0
        try:
            while not self.stop_event.is_set():
                if time.monotonic() - self.start_t >= original.DURATION:
                    break
                ok, frame = self.cap.read()
                now = time.monotonic()
                if not ok or frame is None:
                    continue
                idx = self.frame_count
                elapsed = now - self.start_t
                position = detector(frame, prev, lost, elapsed)
                if position is None:
                    lost += 1
                    if lost >= 4:
                        prev = None
                    self.positions.clear()
                else:
                    self.detected += 1
                    prev = position
                    lost = 0
                    self.positions.append((idx, elapsed, position))
                    if idx in self.pending:
                        row_idx = self.pending.pop(idx)
                        self.shadow_rows[row_idx]['actual_x'] = position[0]
                        self.shadow_rows[row_idx]['actual_y'] = position[1]
                # Same 1.4-2.3s training time domain. Only consecutive detected frames.
                if (position is not None and 1.4 <= elapsed <= 2.3
                        and len(self.positions) == 5
                        and all(self.positions[j+1][0] - self.positions[j][0] == 1 for j in range(4))):
                    hist = np.array([p[2] for p in self.positions], np.float32)
                    feature = np.r_[((hist - hist[-1])/100).reshape(-1), elapsed-self.train_start].astype(np.float32)
                    t0 = time.perf_counter()
                    with torch.inference_mode():
                        input_x = torch.from_numpy(((feature-self.x_mean)/self.x_std)[None, :])
                        delta = self.net(input_x).numpy()[0]*self.y_std+self.y_mean
                    latency_ms = (time.perf_counter()-t0)*1000
                    prediction = hist[-1] + 100*delta
                    row = dict(frame=idx, time_s=round(elapsed,5),
                               observed_x=round(float(hist[-1,0]),2), observed_y=round(float(hist[-1,1]),2),
                               predicted_x=round(float(prediction[0]),2), predicted_y=round(float(prediction[1]),2),
                               target_frame=idx+3, actual_x='', actual_y='',
                               inference_ms=round(latency_ms,3))
                    self.pending[idx+3] = len(self.shadow_rows)
                    self.shadow_rows.append(row)
                    self.predicted += 1
                    self.prediction_ms.append(latency_ms)
                self.writer.write(frame)
                self.frames.append((idx, elapsed))
                self.frame_count += 1
        except Exception as ex:
            self.error = repr(ex)
        finally:
            self.writer.release()
            self.cap.release()

    def stop(self):
        super().stop()
        columns = ['frame','time_s','observed_x','observed_y','predicted_x','predicted_y',
                   'target_frame','actual_x','actual_y','inference_ms']
        with OUT_CSV.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.shadow_rows)
        paired = [r for r in self.shadow_rows if r['actual_x'] != '']
        if paired:
            errors = [np.hypot(r['predicted_x']-r['actual_x'],r['predicted_y']-r['actual_y']) for r in paired]
            print(f'[AI] 実測確認あり: {len(errors)}件 / 平均誤差 {np.mean(errors):.2f}px')
        print(f'[AI] 検出: {self.detected}枚 / 推論: {self.predicted}回 / '
              f'平均推論時間: {np.mean(self.prediction_ms) if self.prediction_ms else 0:.2f}ms')
        print(f'[AI] 保存先: {OUT_CSV}')
        if self.error:
            print(f'[AI] エラー: {self.error}')


if __name__ == '__main__':
    original.FlightRecorder = ShadowRecorder
    original.main()
