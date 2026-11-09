import cv2
import numpy as np
import threading
import time
from flask import Flask, jsonify, render_template_string
import logging

# Disable Flask logging noise
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Shared data
seat_status = [0]*16          # 0=empty(green), 1=occupied(red)
data_lock = threading.Lock()

# Detection thread
def detect_seats():
    global seat_status, empty_count
    cap = cv2.VideoCapture(
        "libcamerasrc ! video/x-raw,width=1920,height=1080,framerate=30/1 ! "
        "videoconvert ! video/x-raw,format=BGR ! appsink",
        cv2.CAP_GSTREAMER
    )
    if not cap.isOpened():
        print("Camera not opened")
        return

    # cv2.namedWindow("Seat Detection", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Seat Detection", 1400, 800)

    rows, cols = 4, 4
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5,5), 0)

        # ---------- layout (same as last version) ----------
        box_size = int(min(w,h)*0.22)
        margin = 30
        inner_gap = 15
        group_gap = 40
        tw = 4*box_size + 2*inner_gap + group_gap
        th = 4*box_size + 2*inner_gap + group_gap
        sx = (w - tw)//2
        sy = (h - th)//2

        new_status = [0]*16
        seat_index = 0
        for r in range(rows):
            for c in range(cols):
                # x offset
                if c==0: xo = 0
                elif c==1: xo = box_size + inner_gap
                elif c==2: xo = 2*box_size + inner_gap + group_gap
                else: xo = 3*box_size + inner_gap + group_gap + inner_gap
                # y offset
                if r==0: yo = 0
                elif r==1: yo = box_size + inner_gap
                elif r==2: yo = 2*box_size + inner_gap + group_gap
                else: yo = 3*box_size + inner_gap + group_gap + inner_gap

                x1 = sx + xo
                y1 = sy + yo
                x2 = x1 + box_size
                y2 = y1 + box_size

                pad = int(box_size*0.25)
                roi = blurred[y1+pad:y2-pad, x1+pad:x2-pad]
                _, thresh = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY_INV)
                area = (x2-x1-2*pad)*(y2-y1-2*pad)
                if area > 0:
                    occ = cv2.countNonZero(thresh) / area
                else:
                    occ = 0
                # occupied if ratio > 0.12
                new_status[seat_index] = 1 if occ > 0.12 else 0
                seat_index += 1

        with data_lock:
            seat_status = new_status.copy()
            empty_count = seat_status.count(0)   # 0 means empty (green)

        time.sleep(0.1)   # small delay to reduce CPU

    cap.release()
    # cv2.destroyAllWindows()

# Start detection thread
detect_thread = threading.Thread(target=detect_seats, daemon=True)
detect_thread.start()

# Flask web server
app = Flask(__name__)

#HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Seat Occupancy</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Segoe UI', Roboto, system-ui, sans-serif; }
        body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; display: flex; justify-content: center; align-items: center; padding: 20px; }
        .card { background: rgba(255,255,255,0.95); backdrop-filter: blur(10px); border-radius: 30px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); padding: 40px; width: fit-content; max-width: 900px; width: 100%; }
        h1 { color: #333; font-weight: 600; margin-bottom: 10px; display: flex; align-items: center; gap: 10px; }
        h1 svg { width: 36px; height: 36px; fill: #667eea; }
        .stats { display: flex; gap: 30px; margin-bottom: 30px; flex-wrap: wrap; }
        .stat-box { background: #f0f4fa; padding: 15px 25px; border-radius: 20px; flex: 1; min-width: 150px; }
        .stat-label { font-size: 14px; color: #555; letter-spacing: 0.5px; }
        .stat-value { font-size: 42px; font-weight: 700; color: #333; line-height: 1.2; }
        .stat-value small { font-size: 16px; font-weight: 400; color: #888; }
        .seat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; justify-items: center; margin-top: 20px; }
        .seat { width: 100px; height: 100px; border-radius: 20px; display: flex; flex-direction: column; justify-content: center; align-items: center; transition: transform 0.2s, box-shadow 0.2s; box-shadow: 0 8px 20px rgba(0,0,0,0.1); cursor: pointer; }
        .seat:hover { transform: scale(1.05); box-shadow: 0 12px 28px rgba(0,0,0,0.2); }
        .seat.empty { background: #4caf50; }
        .seat.occupied { background: #f44336; }
        .seat-number { font-size: 24px; font-weight: 700; color: white; text-shadow: 0 2px 4px rgba(0,0,0,0.2); }
        .seat-label { font-size: 14px; color: rgba(255,255,255,0.9); text-transform: uppercase; letter-spacing: 1px; }
        .footer { margin-top: 30px; text-align: center; color: #888; font-size: 14px; border-top: 1px solid #ddd; padding-top: 20px; }
        .last-updated { font-family: monospace; background: #eef2f6; padding: 5px 15px; border-radius: 50px; display: inline-block; }
        @media (max-width: 600px) {
            .seat { width: 70px; height: 70px; }
            .seat-number { font-size: 18px; }
        }
    </style>
</head>
<body>
    <div class="card">
        <h1>
            <svg viewBox="0 0 24 24"><path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/></svg>
            Seat Occupancy Dashboard
        </h1>
        <div class="stats">
            <div class="stat-box">
                <div class="stat-label">TOTAL SEATS</div>
                <div class="stat-value">16</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">EMPTY</div>
                <div class="stat-value" id="emptyCount">0</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">OCCUPIED</div>
                <div class="stat-value" id="occupiedCount">0</div>
            </div>
        </div>

        <div class="seat-grid" id="seatGrid">
            <!-- seats will be injected by JavaScript -->
        </div>

        <div class="footer">
            <span class="last-updated" id="lastUpdated">Loading...</span>
        </div>
    </div>

    <script>
        async function fetchStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                updateDashboard(data);
            } catch (error) {
                console.error('Error fetching data:', error);
            }
        }

        function updateDashboard(data) {
            document.getElementById('emptyCount').innerText = data.empty;
            document.getElementById('occupiedCount').innerText = 16 - data.empty;

            const grid = document.getElementById('seatGrid');
            grid.innerHTML = '';
            for (let i = 0; i < data.seats.length; i++) {
                const seatDiv = document.createElement('div');
                seatDiv.className = `seat ${data.seats[i] === 0 ? 'empty' : 'occupied'}`;
                seatDiv.innerHTML = `
                    <span class="seat-number">${i+1}</span>
                    <span class="seat-label">${data.seats[i] === 0 ? 'Empty' : 'Occupied'}</span>
                `;
                grid.appendChild(seatDiv);
            }

            document.getElementById('lastUpdated').innerText = 'Updated: ' + new Date().toLocaleTimeString();
        }

        // Initial fetch
        fetchStatus();
        // Refresh every 2 seconds
        setInterval(fetchStatus, 2000);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/status')
def status():
    with data_lock:
        return jsonify({
            'seats': seat_status,
            'empty': empty_count
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
