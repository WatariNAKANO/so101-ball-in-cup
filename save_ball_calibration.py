import json
from pathlib import Path

import cv2
import calibrate_ball_camera as calib

# このスクリプトは既存の校正プログラムを
# importすると実行が始まるため、
# ここでは画像取得を独立して実施します。
