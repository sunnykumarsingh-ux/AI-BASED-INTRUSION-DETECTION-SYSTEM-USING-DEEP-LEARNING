"""
alert_manager.py
----------------
Sirf alert logic — TTS + Telegram + Screenshot + Log.
Drawing ab main.py mein hoti hai taaki cached frames pe bhi boxes dikhen.
"""
import time
import threading
import os
import cv2
from collections import deque, defaultdict
from datetime import datetime
from secure_logger import write_log
 
 
class AlertManager:
    def __init__(self, config: dict):
        self.CRITICAL = [c.lower() for c in config.get("critical_objects", [])]
        self.WARNING   = [w.lower() for w in config.get("warning_objects", [])]
        self.DETECTION_TIMEOUT   = config.get("detection_timeout", 5)
        self.SCREENSHOT_INTERVAL = config.get("screenshot_interval", 10)
        self.OUTPUT_DIR = "detections"
 
        self.last_detected  = defaultdict(float)
        self.last_screenshot = defaultdict(float)
 
        self._speech_queue = deque()
        self._speech_lock  = threading.Lock()
        self._tts_engine   = None
        self._init_tts()
        threading.Thread(target=self._speech_worker, daemon=True).start()
 
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        print("[Alert] Manager ready")
 
    def _init_tts(self):
        try:
            import pyttsx3
            self._tts_engine = pyttsx3.init()
            self._tts_engine.setProperty("rate", 170)
            self._tts_engine.setProperty("volume", 0.8)
            print("[Alert] TTS ready")
        except Exception:
            print("[Alert] TTS unavailable")
 
    def _speech_worker(self):
        while True:
            if self._speech_queue:
                with self._speech_lock:
                    text = self._speech_queue.popleft()
                if self._tts_engine:
                    try:
                        self._tts_engine.say(text)
                        self._tts_engine.runAndWait()
                    except Exception:
                        pass
            time.sleep(0.2)
 
    def _speak(self, text: str):
        with self._speech_lock:
            self._speech_queue.append(text)
 
    def _send_telegram(self, text: str, image_path: str = None):
        def _task():
            try:
                from telegram_alert import send_telegram_alert, send_telegram_photo
                send_telegram_alert(text)
                if image_path:
                    send_telegram_photo(image_path, text)
                print(f"[Alert] Telegram sent: {text[:60]}")
            except Exception as e:
                print(f"[Alert] Telegram error: {e}")
        threading.Thread(target=_task, daemon=True).start()
 
    def get_severity(self, label: str) -> tuple:
        if label in self.CRITICAL:
            return "CRITICAL", (0, 0, 255)
        elif label in self.WARNING:
            return "WARNING", (0, 165, 255)
        return "NORMAL", (0, 255, 255)
 
    def process(self, frame, detections: list):
        """
        Alert logic only — screenshot, TTS, Telegram.
        Drawing ab main.py ke draw_detections() mein hoti hai.
        """
        now = time.time()
 
        for det in detections:
            label     = det["label"]
            conf      = det["conf"]
            x1,y1,x2,y2 = det["bbox"]
            face_name = det.get("face_name", "Unknown")
            age       = det.get("age",    "Unknown")
            gender    = det.get("gender", "Unknown")
            emotion   = det.get("emotion","Unknown")
 
            severity, _ = self.get_severity(label)
 
            if now - self.last_detected[label] > self.DETECTION_TIMEOUT:
                msg = f"{severity}: {label}"
                if face_name != "Unknown":
                    msg += f" — {face_name}"
                self._speak(msg)
 
                if now - self.last_screenshot[label] > self.SCREENSHOT_INTERVAL:
                    folder = os.path.join(self.OUTPUT_DIR, label.replace(" ", "_"))
                    os.makedirs(folder, exist_ok=True)
                    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = os.path.join(folder, f"{label}_{ts}.jpg")
                    cv2.imwrite(path, frame)
                    self.last_screenshot[label] = now
 
                    write_log(datetime.now().isoformat(), label, f"{conf:.2f}",
                              x1, y1, x2, y2, path, severity, face_name)
 
                    if severity == "CRITICAL":
                        # FIX: age/gender/emotion ab cache se aata hai — "Unknown" nahi rahega
                        deepface_line = ""
                        if age != "Unknown":
                            deepface_line = f"\nAge: {age} | Gender: {gender} | Emotion: {emotion}"
                        alert_text = (
                            f"🚨 CRITICAL ALERT\n"
                            f"Object : {label}\n"
                            f"Person : {face_name}"
                            f"{deepface_line}"
                        )
                        self._send_telegram(alert_text, path)
 
                self.last_detected[label] = now
 