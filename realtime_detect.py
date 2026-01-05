import cv2
import datetime
import requests
import time
import os
from dotenv import load_dotenv# .envファイルを読み込む
load_dotenv()

# 変数からURLを取得する
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

# --- ★設定エリア★ ---
# 取得したDiscord Webhook URLをここに貼る
WEBHOOK_URL = "https://discord.com/api/webhooks/XXXXXXX"

# 通知を送った後、次に送るまで何秒待つか（例: 180秒 = 3分）
COOL_DOWN_SECONDS = 900 
# --------------------

print("モデルを読み込んでいます...")
model = YOLO("best.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 録画設定（前回と同じ）
now = datetime.datetime.now()
filename = now.strftime("dog_%Y%m%d_%H%M%S.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(filename, fourcc, 10.0, (640, 480))

# 通知制御用の変数
last_alert_time = 0

print(f"監視開始！通知クールダウンは {COOL_DOWN_SECONDS}秒 です。")

def send_discord_alert(image_array):
    """Discordに画像付きで通知を送る関数"""
    try:
        # 画像を一旦ファイルとして保存せずにメモリ上でエンコード
        _, img_encoded = cv2.imencode('.jpg', image_array)
        
        # 送信データ作成
        files = {
            'file': ('dog_alert.jpg', img_encoded.tobytes(), 'image/jpeg')
        }
        payload = {
            "content": "🚨 **緊急速報** 🚨\nワンちゃんが起きました！"
        }
        
        # 送信
        requests.post(WEBHOOK_URL, data=payload, files=files)
        print("📲 Discordに画像付き通知を送りました")
        
    except Exception as e:
        print(f"❌ 通知エラー: {e}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # AI推論
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    # --- ★通知ロジックここから ---
    # 検出された結果の中にクラスID '1' (Awake) があるか探す
    # ※あなたの環境のIDに合わせてください (0:Sleeping, 1:Awake のはず)
    is_awake = False
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        if cls_id == 1:  # 1がAwakeの場合
            is_awake = True
            break
    
    if is_awake:
        current_time = time.time()
        # 前回の通知から指定時間が経過しているかチェック
        if current_time - last_alert_time > COOL_DOWN_SECONDS:
            print("❗ Awake検知！通知を送信します...")
            send_discord_alert(annotated_frame) # 画像付きで送信
            last_alert_time = current_time # タイマーリセット
    # --- ★通知ロジックここまで ---

    out.write(annotated_frame)
    cv2.imshow("Dog Monitor AI", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
