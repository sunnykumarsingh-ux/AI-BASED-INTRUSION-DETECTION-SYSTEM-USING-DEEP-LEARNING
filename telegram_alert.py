import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==============================
# 🔐 SECURE BOT DETAILS FROM ENV
# ==============================

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TOKEN or not CHAT_ID:
    raise ValueError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in .env file")

# ==============================
# 📡 TELEGRAM ALERT FUNCTION
# ==============================

def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message
    }
    requests.post(url, data=payload)

def send_telegram_photo(photo_path, caption=""):
    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    with open(photo_path, "rb") as photo:
        payload = {
            "chat_id": CHAT_ID,
            "caption": caption
        }
        files = {
            "photo": photo
        }
        requests.post(url, data=payload, files=files)