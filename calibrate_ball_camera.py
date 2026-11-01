import cv2
import json
import numpy as np
from pathlib import Path

DEVICE = "/dev/video0"
OUT = Path("ball_camera_calibration.json")

def candidates(frame):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 小さいノイズを低減
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(
        gray, 65, 255, cv2.THRESH_BINARY_INV
    )

    n, labels, stats, centers = \
        cv2.connectedComponentsWithStats(binary)

    found = []

    for i in range(1, n):
        x, y, bw, bh, area = stats[i]

        if not (100 <= area <= 1800):
            continue
        if not (12 <= bw <= 60 and 12 <= bh <= 60):
            continue
        if not (0.70 <= bw / bh <= 1.30):
            continue

        fill = area / (bw * bh)
        if not (0.50 <= fill <= 0.90):
            continue

        cx, cy = centers[i]

        # 画像端で切れている物体は除外
        if x < 5 or y < 5:
            continue
        if x + bw > w - 5 or y + bh > h - 5:
            continue

        # 周囲が白い背景であることを確認
        margin = 8
        xa = max(0, x - margin)
        ya = max(0, y - margin)
        xb = min(w, x + bw + margin)
        yb = min(h, y + bh + margin)

        patch = gray[ya:yb, xa:xb].copy()
        patch[y-ya:y-ya+bh, x-xa:x-xa+bw] = 0

        outside = np.ones(patch.shape, dtype=bool)
        outside[y-ya:y-ya+bh, x-xa:x-xa+bw] = False

        if not np.any(outside):
            continue

        bright_ratio = np.mean(patch[outside] > 110)
        if bright_ratio < 0.65:
            continue

        found.append({
            "x": float(cx),
            "y": float(cy),
            "width": int(bw),
            "height": int(bh),
            "area": int(area)
        })

    return sorted(found, key=lambda v: v["x"])

cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)
if not cap.isOpened():
    raise RuntimeError("カメラを開けません")

try:
    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 25)

    frame = None

    for _ in range(30):
        ok, img = cap.read()
        if ok and img is not None:
            frame = img

    if frame is None:
        raise RuntimeError("画像を取得できません")

    found = candidates(frame)
    preview = frame.copy()

    print("\n=== ボール候補 ===")

    for i, obj in enumerate(found):
        x = round(obj["x"])
        y = round(obj["y"])

        print(
            f"{i}: X={obj['x']:.1f}, "
            f"Y={obj['y']:.1f}, "
            f"size={obj['width']}x{obj['height']}"
        )

        cv2.circle(preview, (x, y), 20, (0, 255, 0), 2)
        cv2.putText(
            preview, str(i), (x+22, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 0, 255), 2
        )

    cv2.imwrite("ball_calibration_preview.jpg", preview)
    print("確認画像: ball_calibration_preview.jpg")

    if not found:
        raise RuntimeError(
            "ボール候補がありません。画角・背景を確認してください"
        )

    print(
        "\n別のターミナルで画像を開いて確認してください:\n"
        "xdg-open ball_calibration_preview.jpg"
    )
    choice = input(
        "ボールの番号を入力（中止はEnter）: "
    ).strip()

    if choice == "":
        print("校正を中止しました")
        raise SystemExit(0)

    index = int(choice)
    if not (0 <= index < len(found)):
        raise ValueError("候補番号が範囲外です")
    target = found[index]

    # 初回検出したボールの画像をテンプレートとして追跡
    gray0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    tx = round(target["x"])
    ty = round(target["y"])
    r = 19

    template = gray0[ty-r:ty+r, tx-r:tx+r].copy()

    if template.shape != (2*r, 2*r):
        raise RuntimeError("ボールが画像端に近すぎます")

    xs, ys = [], []
    scores = []

    for _ in range(40):
        ok, img = cap.read()
        if not ok or img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 初期位置の周囲80pxを探索
        search_r = 30
        x1 = max(0, tx-search_r)
        y1 = max(0, ty-search_r)
        x2 = min(gray.shape[1], tx+search_r)
        y2 = min(gray.shape[0], ty+search_r)

        search = gray[y1:y2, x1:x2]

        result = cv2.matchTemplate(
            search,
            template,
            cv2.TM_CCOEFF_NORMED
        )

        _, score, _, loc = cv2.minMaxLoc(result)
        scores.append(score)

        if score < 0.55:
            continue

        cx = x1 + loc[0] + r
        cy = y1 + loc[1] + r

        xs.append(float(cx))
        ys.append(float(cy))

    print(
        f"テンプレート一致度: "
        f"最小={min(scores):.3f}, "
        f"最大={max(scores):.3f}"
    )

    print(f"連続検出: {len(xs)}/40")

    if len(xs) < 36:
        raise RuntimeError(
            "検出が不安定です。校正値は保存しません"
        )

    sx = float(np.std(xs))
    sy = float(np.std(ys))

    if max(sx, sy) > 3.0:
        raise RuntimeError(
            f"ボールが揺れています: "
            f"sigmaX={sx:.2f}, sigmaY={sy:.2f}"
        )

    cx = float(np.mean(xs))
    cy = float(np.mean(ys))

    radius = 75
    roi = [
        max(0, int(cx-radius)),
        max(0, int(cy-radius)),
        min(frame.shape[1], int(cx+radius)),
        min(frame.shape[0], int(cy+radius))
    ]

    data = {
        "reference_x": round(cx, 3),
        "reference_y": round(cy, 3),
        "roi": roi,
        "sigma_x": round(sx, 3),
        "sigma_y": round(sy, 3),
        "detected_frames": len(xs),
        "image_size": [
            frame.shape[1], frame.shape[0]
        ]
    }

    OUT.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8"
    )

    print("\n=== 校正完了 ===")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"保存: {OUT}")

finally:
    cap.release()
