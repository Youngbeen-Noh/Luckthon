from flask import Flask, request, jsonify, render_template, send_file
from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path
import datetime

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'  # 이미지 저장 경로를 static/uploads로 설정
analyzed_data = {}
camera_names = {}

# 업로드 폴더가 존재하지 않으면 생성
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 카메라 이름에 따라 일련번호가 붙은 파일명을 생성하는 함수
def get_next_filename(camera_name):
    count = camera_names.get(camera_name, 0)
    filename = f"{camera_name}.jpg"
    camera_names[camera_name] = count + 1
    return filename

@app.route('/')
def index():
    return render_template('Web.html')

@app.route('/add_camera', methods=['POST'])
def add_camera():
    # 새로운 카메라 ID 생성
    camera_id = f"Camera_{len(camera_names) + 1}"
    camera_names[camera_id] = 0  # 초기 일련번호 설정
    analyzed_data[camera_id] = {
        "person_count": 0,
        "timestamp": None,
        "filename": None
    }
    return jsonify({"camera_id": camera_id}), 200

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "이미지가 누락되었습니다.", 400

    image_file = request.files['file']
    image = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # camera_name
    camera_name = Path(image_file.filename).stem
    if camera_name not in camera_names:
        return "아직 카메라가 설치되지 않았습니다.", 400

    # YOLO 모델 로드 및 이미지 분석
    model_name = 'yolo11n'
    model_path = os.path.join(os.path.dirname(__file__), f'model/{model_name}.pt')
    model = YOLO(model_path)
    results = model.predict(image, stream=True, conf=0.1, imgsz=1280)

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
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": output_filename
    }

    return jsonify({"status": "success", "person_count": person_count}), 200

@app.route('/image/<camera_name>', methods=['GET'])
def get_image(camera_name):
    # 카메라 이름을 기반으로 저장된 최신 분석 이미지 반환
    if camera_name in analyzed_data and analyzed_data[camera_name]["filename"]:
        filename = analyzed_data[camera_name]["filename"]
        return send_file(os.path.join(UPLOAD_FOLDER, filename), mimetype='image/jpeg')
    return "해당 카메라의 분석 데이터가 없습니다.", 404


@app.route('/reset_cameras', methods=['POST'])
def reset_cameras():
    # 서버의 분석 데이터 및 파일 초기화
    analyzed_data.clear()
    camera_names.clear()
    for file in os.listdir(UPLOAD_FOLDER):
        os.remove(os.path.join(UPLOAD_FOLDER, file))
    return jsonify({"message": "카메라 데이터 초기화 완료"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
