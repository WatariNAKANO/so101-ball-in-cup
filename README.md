# SO-101 Ball-in-Cup

Physical AI講座2026 応用編1の最終課題です。
SO-101ロボットアームで、紐付きボールを投げ上げてカップで受け止める動作に取り組みました。

人による動作教示を出発点に、機械的な調整、映像による運動解析、初期揺れの影響検証、MLPによる未来位置予測まで実施しました。

## 1. 開発の流れと成果

### 動作教示と固定軌道再生
Leaderで教示した動作を記録し、Followerで再生しました。主にID 2・3・4の3軸を使用しています。
初期実験では、手首角度を約−25度補正した固定軌道で10回中8回成功しました。
これはAIを使用しないオープンループ制御の結果です。

### カップと紐の改良
3Dプリンタによるカップの改造と紐長の調整後、9回連続成功を確認しました。
その後の全試行は未記録のため、集計した成功率ではなく連続成功の実績として示します。
改良前後では機械的な条件が異なります。

### 映像による運動解析
初期揺れの検証に先立ち、4K・120fps映像からボールとカップの位置を自動アノテーションし、両者の軌道をグラフ化しました。
座標CSVとグラフは chapter2_ball/ と chapter2_cup/ に収録しています。
撮影時のフレームレートと解析データのサンプリング間隔は区別して扱ってください。

### 初期揺れの影響検証
安定した再生が得られた後、初期揺れを与えた5試行を行い、初期状態の変化が固定軌道再生に与える影響を検証しました。

### MLPによる未来位置予測
ボール位置の時系列から、MLP（多層パーセプトロン）で約0.12秒先の位置を予測しました。
学習コード、学習済みモデル、座標データ、予測ログ、評価グラフを公開しています。
本提出時点でのAIの役割は未来位置の予測と評価です。予測結果による実機のオンライン軌道補正は今後の課題です。

## 2. ハードウェア・環境

- SO-101 Leader / Follower
- 3Dプリンタで製作・改造したカップ
- ボール：ダンロップ製スカッシュボール（ダブルイエロードット）、紐を取り付けて使用
- ASUS Ascent GX10
- Ubuntu 24.04 / Python 3.12
- USBカメラ
- 運動解析用の4K・120fps撮影映像

主要パッケージの実測バージョンは requirements-observed.txt に記載しています。
これは環境の記録であり、完全なインストール手順ではありません。
pixi.toml / pixi.lock は初期公開時の設定で、追加したカメラ・MLP処理の全依存関係を網羅していません。
PyTorch等は使用するハードウェアに対応した導入が必要です。

## 3. 主なファイル

| ファイル・フォルダ | 内容 |
|---|---|
| teleop_ball_3axis.py | 3軸の動作教示 |
| ball_record_replay.py | 初期公開版の記録・再生 |
| ball_record_replay_id5off_v3.py | カメラ連携で使用する再生本体 |
| ball_throw_D_wrist25_minus.json | カメラ連携処理が参照する成功軌道 |
| trajectories/ | 初期公開時の成功軌道 |
| ball_replay_camera_ready.py | 初期状態の確認と再生 |
| ball_replay_record_flight.py | 再生と飛翔映像の記録 |
| collect_flight_trials.sh | 試行データの保存 |
| calibrate_ball_camera.py / save_ball_calibration.py | カメラ校正関連 |
| ball_camera_calibration.json | 実験環境の校正値 |
| track_flight_ball.py | 飛翔映像のボール追跡 |
| train_ball_predictor.py | MLPの学習 |
| ball_shadow_live.py | 推論と予測ログ記録 |
| analyze_shadow_predictions.py | 予測評価・グラフ化 |
| flight_dataset/trial_*/ | 学習用座標CSV・時刻・試行情報 |
| ball_ai_result/ | 学習済みモデル・テスト予測 |
| shadow_predictions.csv / flight_timestamps.csv | 予測評価用ログ |
| shadow_analysis/ | 比較CSV・評価グラフ |
| chapter2_ball/ / chapter2_cup/ | 映像解析の座標CSV・グラフ |

学習用座標CSVを含む9試行と、初期揺れの検証5試行は別の実験です。

## 4. 解析の実行

必要な依存パッケージを導入し、リポジトリのルートから実行してください。

予測ログの評価：

```bash
python analyze_shadow_predictions.py
```

MLPの再学習：

```bash
python train_ball_predictor.py --data flight_dataset --out ball_ai_result_retrained
```

再学習結果は、公開済みモデルとは別フォルダへ保存します。
学習時の引数や環境によって結果は異なります。

## 5. 実機で使用する場合

シリアルポート、カメラデバイス、校正値、関節設定は実験機固有です。
使用環境に合わせて確認・変更してください。
実機動作前に可動範囲、取り付け、初期姿勢を確認し、直ちに停止できる状態で使用してください。

## 6. 今後の課題

- 初期状態を変えた試行数の拡充
- 未来位置予測の汎化性能の検証
- 予測結果を用いた動作補正
- 補正導入前後の成功率比較

## 7. 動画

[SO-101 Ball-in-Cup 実験動画](https://youtu.be/NdwcPOmxEe4)
