# 🛡️ AI-Based Intrusion Detection System (AI-IDS) 🧬

An advanced, real-time surveillance system powered by **Deep Learning** (YOLOv9) and **Computer Vision**. This project provides automated object detection, facial recognition, and instant multi-channel alerting (Telegram & Voice) with a web dashboard for live monitoring.

![Project Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Python Version](https://img.shields.io/badge/Python-3.9+-blue)
![Deep Learning](https://img.shields.io/badge/YOLO-v8/v9-orange)

## 🌟 Features

- **🚀 YOLOv9 Detection**: Ultra-fast object detection for critical/warning items (phones, weapons, laptops).
- **👤 Facial Recognition**: Integrated face identification to distinguish between known personnel and intruders.
- **📊 Live Web Dashboard**: Real-time video feed and detection metrics via Flask & SocketIO.
- **🚨 Multi-Channel Alerts**:
  - **Telegram**: Instant photos and alerts sent to your private chat.
  - **Voice (TTS)**: Local audio announcements of detected objects.
  - **Secure Logging**: Encrypted recording of all security events.
- **📂 Modular Architecture**: Parallel processing using Python `multiprocessing` to ensure zero lag in video analysis.

## 🛠️ Tech Stack

- **Backend**: Python, OpenCV, Multiprocessing
- **Deep Learning**: YOLOv8/v9 (Ultralytics), DeepFace, Torch
- **Dashboard**: Flask, SocketIO
- **Alerting**: Telegram Bot API, Pyttsx3 (TTS)
- **Data**: CSV, Pickle for Face Data

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have Python 3.9+ installed and a working webcam.

### 2. Installation
```bash
git clone https://github.com/sunnykumarsingh-ux/AI-BASED-INTRUSION-DETECTION-SYSTEM-USING-DEEP-LEARNING.git
cd AI-BASED-INTRUSION-DETECTION-SYSTEM-USING-DEEP-LEARNING
pip install -r requirements.txt
```

### 3. Configuration
Copy `.env.example` to `.env` and add your Telegram credentials:
```bash
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_CHAT_ID=your_id_here
```

### 4. Run the System
```bash
python main.py
```

## 🎮 Controls
- `m`: Clean shutdown
- `r`: Reload facial recognition data

## 📄 License
MIT License - Developed by **Sunny Kumar Singh**

---
*Disclaimer: This system is intended for research purposes. Ensure compliance with local privacy laws.*
