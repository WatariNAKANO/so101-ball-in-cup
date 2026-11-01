#!/usr/bin/env bash

cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1

mkdir -p flight_dataset

# 既存の動画を保護
if [ -f flight_camera.avi ]; then
    stamp=$(date +%Y%m%d_%H%M%S)
    mkdir -p "flight_dataset/backup_$stamp"

    cp flight_camera.avi \
       "flight_dataset/backup_$stamp/"

    for f in flight_timestamps.csv flight_camera_info.txt; do
        if [ -f "$f" ]; then
            cp "$f" "flight_dataset/backup_$stamp/"
        fi
    done

    echo "既存動画をバックアップしました。"
fi

echo "=== SO-101 Flight Data Collection ==="

# 既存の成功軌道・READY判定をそのまま実行
PYTHONNOUSERSITE=1 pixi run python \
    ball_replay_record_flight.py

result=$?

if [ "$result" -ne 0 ]; then
    echo "実行が正常終了していません。"
    echo "今回のデータは保存対象から除外します。"
    exit "$result"
fi

if [ ! -s flight_camera.avi ] ||
   [ ! -s flight_timestamps.csv ]; then
    echo "録画データが見つかりません。"
    exit 1
fi

stamp=$(date +%Y%m%d_%H%M%S)
trial="flight_dataset/trial_$stamp"

mkdir -p "$trial"

cp flight_camera.avi "$trial/"
cp flight_timestamps.csv "$trial/"

if [ -f flight_camera_info.txt ]; then
    cp flight_camera_info.txt "$trial/"
fi

echo
echo "カップインの結果を入力してください。"
read -r -p "成功=s / 失敗=f / 未確認=u : " answer

case "$answer" in
    s|S) outcome="success" ;;
    f|F) outcome="failure" ;;
    *)   outcome="unknown" ;;
esac

{
    echo "trial=$stamp"
    echo "outcome=$outcome"
    echo "trajectory=ball_throw_D_wrist25_minus.json"
    echo "camera=USB_640x480_25fps"
    echo "ready_sigma_limit_px=4.0"
    echo "ready_center_limit_px=10.0"
} > "$trial/trial_info.txt"

echo
echo "=== 保存完了 ==="
echo "保存先: $trial"
echo "結果: $outcome"
