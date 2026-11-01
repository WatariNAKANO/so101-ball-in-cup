import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SOURCE = Path("shadow_predictions.csv")
TIMESTAMPS = Path("flight_timestamps.csv")
OUT = Path("shadow_analysis")
OUT.mkdir(exist_ok=True)

if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

# カメラのフレーム番号と実際の取得時刻
frame_times = {}

if TIMESTAMPS.exists():
    with TIMESTAMPS.open(newline="") as f:
        for row in csv.DictReader(f):
            frame_times[int(row["frame"])] = float(
                row["camera_time_s"]
            )

rows = []

with SOURCE.open(newline="") as f:
    for row in csv.DictReader(f):
        item = {
            "frame": int(row["frame"]),
            "target_frame": int(row["target_frame"]),
            "time_s": float(row["time_s"]),
            "observed_x": float(row["observed_x"]),
            "observed_y": float(row["observed_y"]),
            "predicted_x": float(row["predicted_x"]),
            "predicted_y": float(row["predicted_y"]),
            "inference_ms": float(row["inference_ms"]),
        }

        item["target_time_s"] = frame_times.get(
            item["target_frame"],
            item["time_s"] + 0.12
        )

        if row["actual_x"] and row["actual_y"]:
            item["actual_x"] = float(row["actual_x"])
            item["actual_y"] = float(row["actual_y"])

            item["error_px"] = float(np.hypot(
                item["predicted_x"] - item["actual_x"],
                item["predicted_y"] - item["actual_y"]
            ))
        else:
            item["actual_x"] = np.nan
            item["actual_y"] = np.nan
            item["error_px"] = np.nan

        rows.append(item)

if not rows:
    raise RuntimeError("推論結果がありません")

valid = [
    r for r in rows
    if np.isfinite(r["error_px"])
]

print("\n=== AIオンライン推論の評価 ===")
print(f"推論回数: {len(rows)}")
print(f"実測と比較可能: {len(valid)}")

if valid:
    errors = np.array([r["error_px"] for r in valid])
    print(f"平均誤差: {errors.mean():.2f} px")
    print(f"中央値: {np.median(errors):.2f} px")
    print(f"最大誤差: {errors.max():.2f} px")

latencies = np.array([r["inference_ms"] for r in rows])
print(f"平均推論時間: {latencies.mean():.2f} ms")
print(f"最大推論時間: {latencies.max():.2f} ms")

if frame_times:
    horizons = [
        frame_times[r["target_frame"]]
        - frame_times[r["frame"]]
        for r in rows
        if r["frame"] in frame_times
        and r["target_frame"] in frame_times
    ]
    if horizons:
        print(
            f"予測先までの実時間: "
            f"平均{np.mean(horizons)*1000:.1f} ms"
        )

# 各予測の詳細CSV
with (OUT / "prediction_comparison.csv").open(
    "w", newline=""
) as f:
    cols = [
        "frame", "time_s", "target_frame",
        "target_time_s",
        "predicted_x", "predicted_y",
        "actual_x", "actual_y",
        "error_px", "inference_ms"
    ]
    writer = csv.DictWriter(f, fieldnames=cols)
    writer.writeheader()
    for r in rows:
        writer.writerow({c: r[c] for c in cols})

# 予測位置と実測位置の時系列比較
for axis in ("x", "y"):
    fig, ax = plt.subplots(figsize=(9, 4.5))

    times = [r["target_time_s"] for r in rows]
    predicted = [r[f"predicted_{axis}"] for r in rows]

    ax.plot(
        times, predicted,
        "o-", label="MLP prediction"
    )

    if valid:
        ax.plot(
            [r["target_time_s"] for r in valid],
            [r[f"actual_{axis}"] for r in valid],
            "s--", label="Measured position"
        )

    ax.set_title(
        f"Ball {axis.upper()} position: predicted vs measured"
    )
    ax.set_xlabel("Camera time (s)")
    ax.set_ylabel(f"{axis.upper()} position (px)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUT / f"prediction_{axis}.png",
        dpi=160
    )
    plt.close(fig)

# 誤差の時間変化
if valid:
    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.plot(
        [r["target_time_s"] for r in valid],
        [r["error_px"] for r in valid],
        "o-"
    )
    ax.axhline(
        errors.mean(),
        linestyle="--",
        label=f"Mean = {errors.mean():.2f} px"
    )
    ax.set_title("MLP prediction error")
    ax.set_xlabel("Target camera time (s)")
    ax.set_ylabel("Euclidean error (px)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUT / "prediction_error.png",
        dpi=160
    )
    plt.close(fig)

print("\n=== 時系列評価 ===")
print("時刻[s] | X誤差[px] | Y誤差[px] | 距離誤差[px]")

for r in valid:
    dx = r["predicted_x"] - r["actual_x"]
    dy = r["predicted_y"] - r["actual_y"]

    print(
        f"{r['target_time_s']:6.3f} | "
        f"{dx:+9.2f} | "
        f"{dy:+9.2f} | "
        f"{r['error_px']:11.2f}"
    )

print("\n=== 保存ファイル ===")
for p in sorted(OUT.iterdir()):
    print(p)
