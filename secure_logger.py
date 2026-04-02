"""
secure_logger.py
----------------
Tamper-evident CSV logger with SHA-256 hash per row.
Alert manager is call karta hai — directly process nahi karta.
"""
import csv
import hashlib
import os
from datetime import datetime

LOG_FILE = "detections/detections_log.csv"


def generate_hash(data_string: str) -> str:
    return hashlib.sha256(data_string.encode()).hexdigest()


def write_log(timestamp, label, confidence, x1, y1, x2, y2,
              screenshot, severity="NORMAL", face_name="Unknown"):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    log_data = f"{timestamp},{label},{confidence},{x1},{y1},{x2},{y2},{screenshot},{severity},{face_name}"
    log_hash = generate_hash(log_data)

    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "object", "confidence",
                "x1", "y1", "x2", "y2",
                "screenshot", "severity", "face_name", "hash"
            ])
        writer.writerow([
            timestamp, label, confidence,
            x1, y1, x2, y2,
            screenshot, severity, face_name, log_hash
        ])
