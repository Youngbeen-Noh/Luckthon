from flask import Flask, jsonify, render_template, url_for, request
from flask_socketio import SocketIO, emit
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# 페이지 데이터와 카메라 초기화
page_data = {
    "map_uploaded": False,
    "density_thresholds": {
        "veryCrowded": 10,
        "crowded": 5,
        "moderate": 2,
        "few": 0
    },
    "cameras": []
}
camera_names = {}

@app.route('/')
def index():
    return render_template('userPage.html')

@app.route('/manager')
def manager():
    return render_template('managerPage.html')

@app.route('/get_page_data', methods=['GET'])
def get_page_data():
    page_data["map_url"] = url_for('static', filename='uploads/map.jpg') if page_data["map_uploaded"] else None
    response_data = {
        "page_data": page_data,
        "cameraNames": camera_names,
        "cameraCnt": len(camera_names)
    }
    return jsonify(response_data)

@socketio.on('connect')
def handle_connect():
    print("Client connected")

# 서버에서 클라이언트로 업데이트 전송
def update_clients():
    socketio.emit('update_data', {
        "page_data": page_data,
        "cameraNames": camera_names,
        "cameraCnt": len(camera_names)
    })

# 카메라 추가 시 클라이언트에 업데이트 알림
@app.route('/add_camera', methods=['POST'])
def add_camera():
    camera_id = f"Camera_{len(camera_names) + 1}"
    camera_names[camera_id] = len(camera_names)
    x = request.json.get("x", 100)
    y = request.json.get("y", 200)
    page_data['cameras'].append({"id": camera_id, "x": x, "y": y})
    update_clients()
    return jsonify({"camera_id": camera_id}), 200

# 설정 저장 시 클라이언트에 업데이트 알림
@app.route('/save_page_data', methods=['POST'])
def save_page_data():
    density_thresholds = json.loads(request.form.get("density_thresholds", "{}"))
    page_data["density_thresholds"].update(density_thresholds)
    cameras = json.loads(request.form.get("cameras", "[]"))
    page_data["cameras"] = cameras
    update_clients()
    return jsonify({"status": "success", "message": "Settings and data saved successfully"}), 200

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8000)
