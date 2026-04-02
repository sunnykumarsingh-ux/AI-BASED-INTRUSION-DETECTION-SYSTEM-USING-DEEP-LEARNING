"""
camera_process.py
-----------------
Sirf camera se frames capture karta hai.
Frame queue mein dalta hai — YOLO ka wait nahi karta.
"""
import cv2
import time
from multiprocessing import Process, Queue


def camera_worker(frame_queue: Queue, stop_event):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("[Camera] Started")

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("[Camera] Read failed, retrying...")
            time.sleep(0.5)
            continue

        # Queue full hone par purana frame drop karo — fresh frame zyada zaroori hai
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except Exception:
                pass

        frame_queue.put(frame)

    cap.release()
    print("[Camera] Stopped")
