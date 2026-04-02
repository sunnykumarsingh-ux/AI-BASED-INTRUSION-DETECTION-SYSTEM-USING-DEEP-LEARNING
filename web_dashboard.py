"""
web_dashboard.py
----------------
Simple web dashboard for real-time surveillance monitoring.
Uses Flask-SocketIO for live updates without heavy polling.
"""

import cv2
import base64
import threading
import time
import logging
from flask import Flask, render_template_string, request
from flask_socketio import SocketIO, emit
from typing import Dict, Any, List
import numpy as np

logger = logging.getLogger(__name__)

class WebDashboard:
    def __init__(self, config: Dict[str, Any]):
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.config = config
        self.latest_frame = None
        self.latest_detections = []
        self.is_running = False

        # Routes
        @self.app.route('/')
        def index():
            return render_template_string(self.get_html_template())

        @self.socketio.on('connect')
        def handle_connect():
            logger.info("Client connected to dashboard")
            emit('status', {'message': 'Connected to surveillance system'})

        @self.socketio.on('request_update')
        def handle_update_request():
            self.send_update()

    def get_html_template(self) -> str:
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Surveillance Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #333; color: white; padding: 20px; border-radius: 8px; }
        .stats { display: flex; gap: 20px; margin: 20px 0; }
        .stat { background: white; padding: 15px; border-radius: 8px; flex: 1; text-align: center; }
        .video-section { background: white; padding: 20px; border-radius: 8px; margin: 20px 0; }
        #videoFeed { max-width: 100%; border: 2px solid #333; }
        .alerts { background: white; padding: 20px; border-radius: 8px; max-height: 300px; overflow-y: auto; }
        .alert { padding: 10px; margin: 5px 0; border-left: 4px solid #ff6b6b; background: #fff5f5; }
        .alert.warning { border-left-color: #ffa726; background: #fff8e1; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Real-Time Surveillance Dashboard</h1>
            <p>Live monitoring and alerts</p>
        </div>

        <div class="stats">
            <div class="stat">
                <h3>Status</h3>
                <div id="status">Connecting...</div>
            </div>
            <div class="stat">
                <h3>Detections</h3>
                <div id="detectionCount">0</div>
            </div>
            <div class="stat">
                <h3>Alerts</h3>
                <div id="alertCount">0</div>
            </div>
        </div>

        <div class="video-section">
            <h2>Live Feed</h2>
            <img id="videoFeed" src="" alt="Live video feed" />
        </div>

        <div class="alerts">
            <h2>Recent Alerts</h2>
            <div id="alertsList"></div>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.js"></script>
    <script>
        const socket = io();
        let alertCount = 0;

        socket.on('status', function(data) {
            document.getElementById('status').textContent = data.message;
        });

        socket.on('update', function(data) {
            // Update video feed
            if (data.frame) {
                document.getElementById('videoFeed').src = 'data:image/jpeg;base64,' + data.frame;
            }

            // Update detection count
            document.getElementById('detectionCount').textContent = data.detections ? data.detections.length : 0;

            // Update alerts
            if (data.alerts) {
                const alertsList = document.getElementById('alertsList');
                data.alerts.forEach(alert => {
                    const alertDiv = document.createElement('div');
                    alertDiv.className = 'alert ' + (alert.severity === 'CRITICAL' ? '' : 'warning');
                    alertDiv.innerHTML = `<strong>${alert.timestamp}</strong>: ${alert.message}`;
                    alertsList.insertBefore(alertDiv, alertsList.firstChild);
                    alertCount++;
                });
                document.getElementById('alertCount').textContent = alertCount;
            }
        });

        // Request updates periodically
        setInterval(() => {
            socket.emit('request_update');
        }, 1000);  // Update every second to reduce load
    </script>
</body>
</html>
        """

    def update_frame(self, frame: np.ndarray, detections: List[Dict[str, Any]]):
        """Update the latest frame and detections for dashboard"""
        self.latest_frame = frame
        self.latest_detections = detections

    def send_update(self):
        """Send current data to connected clients"""
        if self.socketio:
            update_data = {
                'detections': self.latest_detections,
                'alerts': []  # Could be populated from alert manager
            }

            if self.latest_frame is not None:
                # Encode frame as base64 for web transmission
                try:
                    _, buffer = cv2.imencode('.jpg', self.latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    update_data['frame'] = base64.b64encode(buffer).decode('utf-8')
                except Exception as e:
                    logger.error(f"Error encoding frame: {e}")

            self.socketio.emit('update', update_data)

    def start(self, host='0.0.0.0', port=5000):
        """Start the web dashboard server"""
        self.is_running = True
        logger.info(f"Starting web dashboard on http://{host}:{port}")
        self.socketio.run(self.app, host=host, port=port, debug=False)

    def stop(self):
        """Stop the dashboard"""
        self.is_running = False
        logger.info("Web dashboard stopped")