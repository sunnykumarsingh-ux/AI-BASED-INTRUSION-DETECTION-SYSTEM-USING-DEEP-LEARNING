"""
main.py — Orchestrator
----------------------
Sare processes shuru karta hai aur unhe connect karta hai.
Queues ke through data flow hota hai — koi bhi process doosre ko block nahi karta.
 
Architecture:
  [camera_process] --(frame_queue)--> [yolo_process]
                                           |
                                    (detection_queue)
                                           |
                                    [face_process]
                                           |
                                    (result_queue)
                                           |
                                    [main loop: display + alert]
"""
 
import cv2
import time
import json
import gc
import torch
import os
import logging
import numpy as np
from typing import Dict, Any, List
from dotenv import load_dotenv
from multiprocessing import Process, Queue, Event
import threading

from camera_process import camera_worker
from yolo_process import yolo_worker
from face_process import face_worker
from alert_manager import AlertManager
from web_dashboard import WebDashboard


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("surveillance.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("surveillance.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    load_dotenv()  # Load .env file
    try:
        # Try loading from .env first, fallback to config.json
        config = {
            "confidence_threshold": float(os.getenv("CONFIDENCE_THRESHOLD", "0.5")),
            "detection_timeout": int(os.getenv("DETECTION_TIMEOUT", "5")),
            "screenshot_interval": int(os.getenv("SCREENSHOT_INTERVAL", "10")),
            "yolo_skip_frames": int(os.getenv("YOLO_SKIP_FRAMES", "3")),
            "critical_objects": [obj.strip() for obj in os.getenv("CRITICAL_OBJECTS", "cell phone,scissors,pen,laptop,weapon").split(",")],
            "warning_objects": [obj.strip() for obj in os.getenv("WARNING_OBJECTS", "book,tablet").split(",")],
        }
        logger.info("Config loaded from environment")
        return config
    except Exception as e:
        logger.error(f"Error loading config from env: {e}, falling back to config.json")
        try:
            with open("config.json") as f:
                return json.load(f)
        except Exception:
            logger.warning("config.json not found — using defaults")
            return {
                "confidence_threshold": 0.5,
                "detection_timeout": 5,
                "screenshot_interval": 10,
                "critical_objects": ["cell phone", "scissors", "pen", "laptop", "weapon"],
                "warning_objects": ["book", "tablet"],
                "yolo_skip_frames": 3
            }
 
 
def draw_detections(frame: np.ndarray, detections: List[Dict[str, Any]], alert_mgr: AlertManager) -> np.ndarray:
    """
    Frame pe boxes draw karo — alert_mgr.process se alag kiya
    taaki cached detections bhi draw ho sakein bina alert trigger kiye.
    """
    for det in detections:
        label = det["label"]
        conf  = det["conf"]
        x1, y1, x2, y2 = det["bbox"]
        face_name = det.get("face_name", "Unknown")
        age    = det.get("age",    "Unknown")
        gender = det.get("gender", "Unknown")
 
        _, color = alert_mgr.get_severity(label)
 
        # Object bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.1f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
 
        # Face bounding box (green) + naam
        if "face_bbox" in det:
            fx, fy, fw, fh = det["face_bbox"]
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
            label_text = face_name
            if age != "Unknown":
                label_text += f"  {age}yo {gender}"
            cv2.putText(frame, label_text,
                        (fx, fy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return frame
 
 
def main():
    config = load_config()
 
    # Queues — maxsize se stale frames drop hote hain automatically
    frame_queue     = Queue(maxsize=4)
    detection_queue = Queue(maxsize=4)
    result_queue    = Queue(maxsize=4)
    # 'r' key ka signal face_proc tak pahunchane ke liye dedicated queue
    reload_queue    = Queue(maxsize=2)
 
    stop_event = Event()
 
    # --- Processes start karo ---
    cam_proc = Process(
        target=camera_worker,
        args=(frame_queue, stop_event),
        name="CameraProcess",
        daemon=True
    )
    yolo_proc = Process(
        target=yolo_worker,
        args=(frame_queue, detection_queue, config, stop_event),
        name="YOLOProcess",
        daemon=True
    )
    face_proc = Process(
        target=face_worker,
        args=(detection_queue, result_queue, reload_queue, stop_event),
        name="FaceProcess",
        daemon=True
    )
 
    cam_proc.start()
    yolo_proc.start()
    face_proc.start()
 
    logger.info("All processes started")
    logger.info("Controls: 'm' = exit | 'r' = reload face data")
    logger.info("=" * 50)
 
    alert_mgr = AlertManager(config)
 
    # Dashboard initialization
    dashboard = WebDashboard(config)
    dashboard_thread = threading.Thread(target=dashboard.start, daemon=True)
    dashboard_thread.start()

    frame_count = 0
    start_time = time.time()
    display_frame = None        # Latest annotated frame for display
 
    # FIX Issue 4: Last known detections cache
    # Skipped frames pe bhi rectangles dikhte rahein
    last_detections = []
    DETECTION_DISPLAY_TTL = 1.5  # seconds — itne time tak cached boxes dikhenge
 
    last_raw_frame = None       # Latest raw camera frame
    last_result_ts = 0.0        # Last result aane ka time
 
    while True:
        try:
            got_new = False
            try:
                data = result_queue.get_nowait()
                raw_frame     = data["frame"].copy()
                last_raw_frame = data["frame"].copy()
                detections    = data["detections"]
                last_detections = detections
                last_result_ts  = time.time()
                got_new = True
                frame_count += 1
 
                # Alerts sirf naye results pe trigger karo
                alert_mgr.process(raw_frame, detections)
 
            except Exception:
                pass  # Queue khali — normal hai
 
            # Display frame prepare karo
            now = time.time()
            if got_new and last_raw_frame is not None:
                display_frame = last_raw_frame.copy()
                # Cached detections draw karo
                if (now - last_result_ts) < DETECTION_DISPLAY_TTL:
                    display_frame = draw_detections(display_frame, last_detections, alert_mgr)
            elif last_raw_frame is not None and (now - last_result_ts) < DETECTION_DISPLAY_TTL:
                # Naya result nahi aaya lekin cached detections abhi valid hain
                display_frame = last_raw_frame.copy()
                display_frame = draw_detections(display_frame, last_detections, alert_mgr)
 
            if display_frame is not None:
                fps = frame_count / max(1, now - start_time)
                cv2.putText(display_frame, f"FPS: {fps:.1f} | Faces: {len(last_detections)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("AI Detection System  [m=exit  r=reload faces]", display_frame)

                # Update web dashboard with current frame and detections
                dashboard.update_frame(display_frame, last_detections)
 
            key = cv2.waitKey(1) & 0xFF
            if key == ord("m"):
                logger.info("Shutting down...")
                break
            elif key == ord("r"):
                # FIX Issue 3: reload signal queue ke through face_proc ko bhejo
                try:
                    reload_queue.put_nowait("reload")
                    logger.info("Reload signal sent to face process")
                except Exception:
                    pass
 
            # Memory cleanup
            if frame_count % 300 == 0 and frame_count > 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
 
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            time.sleep(0.5)
 
    # --- Cleanup ---
    stop_event.set()
    cam_proc.join(timeout=3)
    yolo_proc.join(timeout=5)
    face_proc.join(timeout=5)
    cv2.destroyAllWindows()
 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
 
    logger.info("System stopped cleanly.")
 
 
if __name__ == "__main__":
    main()
 