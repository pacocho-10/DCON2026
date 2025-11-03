"""
line_tracking.py

用途:
- 学習済みU-Netモデルを使ってライン検出
- マスクからライン中心を計算
- 偏差(error)をPID制御に利用可能
- カメラ画像またはテスト画像を使用可能
"""

import cv2
import numpy as np
import tensorflow as tf

# ==========================
# パラメータ
# ==========================
MODEL_PATH = '../light_unet_line_detector.h5'  # 学習済みモデル
CAMERA_INDEX = 0                            # 内蔵カメラ/USBカメラ番号
THRESHOLD = 0.5                             # マスク二値化のしきい値
Kp = 0.002                                  # PIDの比例ゲイン（簡易制御）

# ==========================
# モデル読み込み
# ==========================
model = tf.keras.models.load_model(MODEL_PATH)
input_size = model.input_shape[1:3]  # モデル入力サイズ

# ==========================
# 推論用関数
# ==========================
def preprocess(img):
    """グレースケール→リサイズ→正規化"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
    resized = cv2.resize(gray, input_size)
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=(0,-1))

def predict_mask(img):
    """U-Netでマスク予測"""
    pre = preprocess(img)
    pred = model.predict(pre)[0,:,:,0]
    mask = (pred > THRESHOLD).astype(np.uint8)
    return mask

def compute_error(mask):
    """マスクからライン中心偏差を計算"""
    M = cv2.moments(mask)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
    else:
        cx = mask.shape[1] // 2
    center_x = mask.shape[1] // 2
    error = cx - center_x
    return error

def compute_steer(error):
    """簡易PID制御(比例のみ)"""
    return Kp * error

# ==========================
# カメラまたは画像でのライン追跡
# ==========================
def run_camera_tracking():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("カメラを開けません")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask = predict_mask(frame)
        error = compute_error(mask)
        steer = compute_steer(error)

        # mask をフレームサイズに合わせる
        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

        # 直進/左折/右折判定
        if abs(error) < 5:
            direction = "Straight"
        elif error > 5:
            direction = "Left"
        else:
            direction = "Right"

        # 結果表示
        overlay = cv2.addWeighted(frame, 0.8, cv2.cvtColor(mask_resized*255, cv2.COLOR_GRAY2BGR), 0.5, 0)
        cv2.putText(overlay, f"Error: {error}, Steer: {steer:.3f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
        cv2.putText(overlay, f"Direction: {direction}", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 2)
        cv2.imshow("Line Tracking", overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def run_test_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("画像を読み込めません")
        return
    mask = predict_mask(img)
    error = compute_error(mask)
    steer = compute_steer(error)

    overlay = cv2.addWeighted(img, 0.8, cv2.cvtColor(mask*255, cv2.COLOR_GRAY2BGR), 0.5, 0)
    cv2.putText(overlay, f"Error: {error}, Steer: {steer:.3f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
    cv2.imshow("Test Line Tracking", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ==========================
# メイン
# ==========================
if __name__ == "__main__":
    mode = input("Camera or TestImage? [c/t]: ").strip().lower()
    if mode == 'c':
        run_camera_tracking()
    elif mode == 't':
        path = input("画像パスを入力: ").strip()
        run_test_image(path)
    else:
        print("c または t を入力してください")