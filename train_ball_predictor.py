#!/usr/bin/env python3
"""SO-101 ball flight: baseline and PyTorch MLP 3-frame-ahead prediction.

Usage:
  python train_ball_predictor.py --data flight_tracking
  python train_ball_predictor.py --data flight_dataset --epochs 400

Every trial must contain ball_positions.csv. Trial-wise test split prevents
leaking near-identical frames from the same trajectory into evaluation.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

AP = argparse.ArgumentParser()
AP.add_argument('--data', default='flight_dataset')
AP.add_argument('--out', default='ball_ai_result')
AP.add_argument('--window', type=int, default=5)
AP.add_argument('--horizon', type=int, default=3)
AP.add_argument('--start', type=float, default=1.40)
AP.add_argument('--end', type=float, default=2.30)
AP.add_argument('--epochs', type=int, default=400)
AP.add_argument('--seed', type=int, default=42)
args = AP.parse_args()

root = Path(args.data)
paths = ([root] if (root / 'ball_positions.csv').exists() else
         sorted({p.parent for p in root.rglob('ball_positions.csv')}))
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)


def load_windows(path):
    with (path / 'ball_positions.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    by_frame = {}
    for row in rows:
        try:
            if row['status'] != 'detected':
                continue
            k = int(row['frame'])
            t = float(row['time_s'])
            if args.start <= t <= args.end:
                by_frame[k] = (t, float(row['x_px']), float(row['y_px']))
        except (KeyError, ValueError):
            continue
    samples = []
    for end_i in sorted(by_frame):
        history_idx = list(range(end_i - args.window + 1, end_i + 1))
        future_i = end_i + args.horizon
        if future_i not in by_frame or any(i not in by_frame for i in history_idx):
            continue  # no interpolation or filling missing frames
        hist = np.array([by_frame[i][1:] for i in history_idx], dtype=np.float32)
        target = np.array(by_frame[future_i][1:], dtype=np.float32)
        t0 = by_frame[end_i][0]
        t1 = by_frame[future_i][0]
        if not (0.01 < t1 - t0 < 0.30):
            continue
        # input: relative XY history (last point = 0) and observation time
        feature = np.r_[((hist - hist[-1]) / 100).reshape(-1), t0 - args.start]
        delta = (target - hist[-1]) / 100
        # baselines use actual elapsed time
        dt = (t1 - t0) / (by_frame[end_i][0] - by_frame[end_i-1][0])
        cv = hist[-1] + (hist[-1] - hist[-2]) * dt
        if len(hist) >= 3:
            v = hist[-1] - hist[-2]
            a = hist[-1] - 2*hist[-2] + hist[-3]
            ca = hist[-1] + v*dt + a*(dt*(dt+1)/2)
        else:
            ca = cv
        samples.append({
            'trial': path.name, 'frame': end_i, 'time': t0,
            'feature': feature.astype(np.float32),
            'last': hist[-1], 'target': target, 'delta': delta,
            'cv': cv, 'ca': ca,
        })
    return samples

all_by_trial = [(p, load_windows(p)) for p in paths]
all_by_trial = [(p, s) for p, s in all_by_trial if s]
if not all_by_trial:
    raise SystemExit('有効な飛翔データがありません。CSVに連続した検出区間が必要です。')
trials = {p.name: s for p, s in all_by_trial}
print('=== データ概要 ===')
for name, samples in trials.items():
    print(f'{name}: {len(samples)} 個の入力/正解ペア')


def report(name, samples, pred_key=None, pred=None):
    truth = np.array([s['target'] for s in samples])
    if pred is None:
        pred = np.array([s[pred_key] for s in samples])
    e = np.linalg.norm(pred - truth, axis=1)
    print(f'{name:>20}: MAE(距離)={e.mean():.2f}px, 中央値={np.median(e):.2f}px, 最大={e.max():.2f}px, N={len(e)}')
    return float(e.mean()), e

samples = [s for group in trials.values() for s in group]
report('等速度モデル(全試行)', samples, 'cv')
report('等加速度モデル(全試行)', samples, 'ca')

if len(trials) < 3:
    print('\nAI学習は保留: 独立した飛翔試行が3本以上必要です。')
    print('この結果は教師データ作成の確認であり、未知の試行への予測精度ではありません。')
    raise SystemExit(0)

try:
    import torch
    from torch import nn
except ImportError as ex:
    raise SystemExit(f'PyTorchが見つかりません: {ex}')

torch.manual_seed(args.seed)
np.random.seed(args.seed)
keys = sorted(trials)
# Last trial reserved for final test; validation is a distinct trial when >=4.
test_key = keys[-1]
val_key = keys[-2] if len(keys) >= 4 else None
train_keys = keys[:-2] if val_key else keys[:-1]
train_s = [s for k in train_keys for s in trials[k]]
val_s = trials[val_key] if val_key else []
test_s = trials[test_key]
print(f'\n分割: train={train_keys}, validation={val_key}, test={test_key}')
print('※試行数が少ないため、これは探索的な結果です。')

X = np.array([s['feature'] for s in train_s], dtype=np.float32)
Y = np.array([s['delta'] for s in train_s], dtype=np.float32)
mu = X.mean(0)
sd = np.maximum(X.std(0), 1e-3)
Ym = Y.mean(0)
Ys = np.maximum(Y.std(0), 1e-3)

def to_tensor(items):
    x = np.array([s['feature'] for s in items], np.float32)
    y = np.array([s['delta'] for s in items], np.float32)
    return torch.from_numpy((x - mu)/sd), torch.from_numpy((y - Ym)/Ys)

Xt, Yt = to_tensor(train_s)
Xv, Yv = to_tensor(val_s) if val_s else (None, None)
net = nn.Sequential(nn.Linear(X.shape[1], 64), nn.Tanh(), nn.Linear(64,64), nn.Tanh(), nn.Linear(64,2))
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-3)
lossfn = nn.MSELoss()
best = float('inf')
best_state = None
stop_after = 100
best_epoch = 0
for epoch in range(args.epochs):
    net.train()
    optimizer.zero_grad()
    loss = lossfn(net(Xt), Yt)
    loss.backward()
    optimizer.step()
    net.eval()
    with torch.no_grad():
        criterion = float(lossfn(net(Xv), Yv)) if Xv is not None else float(loss)
    if criterion < best - 1e-5:
        best = criterion
        best_epoch = epoch
        best_state = {k:v.detach().clone() for k,v in net.state_dict().items()}
    if epoch - best_epoch > stop_after:
        break
net.load_state_dict(best_state)
net.eval()

def predict(items):
    with torch.no_grad():
        x,_ = to_tensor(items)
        delta = net(x).numpy()*Ys+Ym
    return np.array([s['last'] for s in items]) + 100*delta

report('等速度(未知試行)',test_s,'cv')
report('等加速度(未知試行)',test_s,'ca')
pred = predict(test_s)
err, _ = report('MLP(未知試行)',test_s,pred=pred)
with (out/'test_predictions.csv').open('w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['trial','frame','time_s','true_x','true_y','pred_x','pred_y','const_vel_x','const_vel_y'])
    for s,p in zip(test_s,pred):
        w.writerow([s['trial'],s['frame'],round(s['time'],3),*map(float,s['target']),*map(float,p),*map(float,s['cv'])])

# Store model and scaling together for later shadow-mode inference.
torch.save({'state_dict':net.state_dict(), 'x_mean':mu, 'x_std':sd,
            'y_mean':Ym,'y_std':Ys, 'window':args.window,
            'horizon':args.horizon,'start':args.start,'end':args.end,
            'train_trials':train_keys,'test_trial':test_key},out/'ball_predictor.pt')
print(f'保存: {out}/ball_predictor.pt, {out}/test_predictions.csv')
