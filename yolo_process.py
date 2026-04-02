"""
yolo_process.py
---------------
Sirf YOLO object detection karta hai.
Har 3rd frame process karta hai — load reduce hota hai.
Result queue mein detection data dalta hai.
"""
import time
import torch
import gc
import logging
from typing import Dict, Any
from ultralytics import YOLO
from multiprocessing import Queue

logger = logging.getLogger(__name__)


def yolo_worker(frame_queue: Queue, detection_queue: Queue, config: Dict[str, Any], stop_event) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Try to load YOLOv9 or fallback to v8
    try:
        model = YOLO("yolov9t.pt")  # Smaller, faster model
        logger.info("Loaded YOLOv9 model")
    except Exception:
        model = YOLO("yolov8n.pt")
        logger.info("Loaded YOLOv8 model")

    model.fuse()

    CONFIDENCE_THRESHOLD = config.get("confidence_threshold", 0.5)
    SKIP_FRAMES = config.get("yolo_skip_frames", 3)   # Har 3rd frame hi process karo

    frame_count = 0
    print("[YOLO] Started")

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1.0)
        except Exception:
            continue

        frame_count += 1

        # Frame skipping — load balance ka sabse important trick
        if frame_count % SKIP_FRAMES != 0:
            # Annotated frame bina detection ke bhi bhejo (display ke liye)
            detection_queue.put({
                "frame": frame,
                "detections": [],
                "skip": True
            })
            continue

        results = model.predict(frame, imgsz=320, device=device, verbose=False, half=True)[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id].lower()
            conf = float(box.conf[0])

            if conf < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "label": label,
                "conf": conf,
                "bbox": (x1, y1, x2, y2)
            })

        detection_queue.put({
            "frame": frame,
            "detections": detections,
            "skip": False
        })

        # Memory cleanup
        if frame_count % 300 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.info("Stopped")
