from flask import Flask, jsonify, render_template, url_for, send_file, request
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import json
import cv2
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
UPLOAD_FOLDER = 'static/uploads'  # 이미지 저장 경로를 static/uploads로 설정
MAP_IMAGE_PATH = os.path.join(UPLOAD_FOLDER, 'map.jpg')


analyzed_data={}
# 페이지 데이터와 카메라 초기화
page_data = {
    "map_uploaded": False,
    "map_url": MAP_IMAGE_PATH,
    "density_thresholds": {
        "veryCrowded": 10,
        "crowded": 5,
        "moderate": 2,
        "few": 0
    },
    "cameras": []
}
camera_names = {}

# 업로드 폴더가 존재하지 않으면 생성
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 카메라 이름에 따라 일련번호가 붙은 파일명을 생성하는 함수
def get_next_filename(camera_name):
    global camera_names
    filename = f"{camera_name}.jpg"
    return filename

# 서버에서 클라이언트로 업데이트 전송
def update_clients():
    global analyzed_data
    print(analyzed_data)
    socketio.emit('update_data', {
        "page_data": page_data,
        "cameraNames": camera_names,
        "analyzed_data": analyzed_data
    })

@app.route('/')
def index():
    return render_template('userPage.html')

@app.route('/manager')
def manager():
    return render_template('managerPage.html')

@socketio.on('connect')
def handle_connect():
    print("Client connected")

# 설정 저장 시 클라이언트에 업데이트 알림
@app.route('/save_page_data', methods=['POST'])
def save_page_data():
    # 지도 저장
    if 'map' in request.files:
        map_file = request.files['map']
        map_file.save(MAP_IMAGE_PATH)
        page_data["map_uploaded"] = True
    else:
        return jsonify({"status" : "map load fail"})

    # 혼잡도 저장
    density_thresholds = json.loads(request.form.get("density_thresholds", "{}"))
    page_data["density_thresholds"].update(density_thresholds)

    # 카메라 포인트 저장
    cameras = json.loads(request.form.get("cameras", "[]"))
    page_data["cameras"] = cameras
    global camera_names
    camera_names = [camera["id"] for camera in page_data["cameras"]]

    global analyzed_data
    analyzed_data = json.loads(request.form.get("analyzed_data"))

    update_clients()

    return jsonify({"status": "success", "message": "Settings and data saved successfully"}), 200

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "이미지가 누락되었습니다.", 400

    global analyzed_data
    global camera_names
    image_file = request.files['file']
    image = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # camera_name
    camera_name = Path(image_file.filename).stem
    if camera_name not in camera_names:
        return "아직 카메라가 설치되지 않았습니다.", 401

    # YOLO 모델 로드 및 이미지 분석
    model_name = 'yolo11n'
    model_path = os.path.join(os.path.dirname(__file__), f'model/{model_name}.pt')
    model = YOLO(model_path)
    results = model.predict(image, stream=True, conf=0.3, imgsz=1280)

    # 분석된 결과에서 사람 수 카운트 및 바운딩 박스 그리기
    for result in results:
        boxes = result.boxes
    person_index = np.where(boxes.cls.cpu().numpy() == 0)[0]
    person_count = len(person_index)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        label = result.names[int(box.cls)]
        confidence = box.conf[0]
        
        # 사람으로 인식된 객체에만 바운딩 박스 추가
        if label == 'person':
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'{label} {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
                # 파일 이름을 생성하여 분석 이미지 저장
    output_filename = get_next_filename(camera_name)
    output_path = os.path.join(UPLOAD_FOLDER, output_filename)
    cv2.imwrite(output_path, image)

    analyzed_data[camera_name] = {
        "person_count": person_count,
        "filename": output_filename
    }

    # 분석 끝나면 정보 업데이트
    update_clients()

    return jsonify({"status": "success", "person_count": person_count}), 200

@app.route('/image/<camera_name>', methods=['GET'])
def get_image(camera_name):
    global analyzed_data
    # 카메라 이름을 기반으로 저장된 최신 분석 이미지 반환
    if camera_name in analyzed_data and analyzed_data[camera_name]["filename"]:
        filename = analyzed_data[camera_name]["filename"]
        return send_file(os.path.join(UPLOAD_FOLDER, filename), mimetype='image/jpeg')
    return "해당 카메라의 분석 데이터가 없습니다.", 404

@app.route('/get_initial_page', methods=['GET'])
def get_page():
    update_clients()
    return ({"status": "success"})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8000)
