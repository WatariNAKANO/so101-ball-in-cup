# SO-101 Ball-in-Cup

Physical AI応用編1の最終課題として、SO-101ロボットアームによる
Ball-in-Cup（ボールを投げ上げ、カップで受け止める動作）に取り組んでいます。

## 1. Project Overview

本プロジェクトでは、以下の方法でBall-in-Cupを実現します。

1. SO-101 Leaderによる動作教示
2. SO-101 Followerによる記録軌道の再生
3. カメラによるボールの初期状態の観測
4. AIによる動作補正
5. AI導入前後の成功率比較

## 2. Hardware

- SO-101 Leader（黒）
- SO-101 Follower（白）
- ASUS Ascent GX10
- Ubuntu 24.04

## 3. Software

- Python
- LeRobot
- scservo_sdk
- pixi

## 4. Current Progress

Leaderで記録した動作をFollowerで再生し、
固定軌道によるBall-in-Cupに成功しました。

手首角度を約25度補正した軌道では、
10回中8回の成功を確認しています。

この結果はAIを使用しない固定軌道制御によるものです。

## 5. Main Files

- `teleop_ball_3axis.py` : 3軸テレオペレーション
- `ball_record_replay.py` : 動作の記録と再生
- `trajectories/ball_throw_D_wrist25_minus.json` : 成功軌道

## 6. Future Development

- 繰り返し試行の自動化
- カメラによる初期揺れの計測
- AIによる動作補正
- 外乱条件での成功率比較

## 7. Notes

ロボットのポート設定やキャリブレーション値は
使用する環境に応じて調整する必要があります。
