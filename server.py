from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import cv2
import numpy as np
import os
import datetime
from pathlib import Path
from waitress import serve

app = Flask(__name__)
analyzed_data = None  # 웹페이지에 표시할 분석 데이터
camera_names=[]

@app.route('/upload', methods=['POST'])
def upload_image():
    global analyzed_data
    global camera_names
    
    # 이미지 파일 확인
    if 'image' not in request.files:
        return "이미지가 없습니다.", 400
    
    # 이미지 변환
    image_file = request.files['image']
    image = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    # 카메라 이름이 배열에 없으면 추가
    camera_name=Path(image_file.filename).stem
    if camera_name not in camera_names:
        camera_names.append(camera_name)
    camera_index = camera_names.index(camera_name)
    
    # YOLO 모델 로드 및 이미지 분석
    model_name = 'yolo11n'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, f'model/{model_name}.pt')

    # 모델 설정 및 분석 시작
    model = YOLO(model_path)
    results = model.predict(image, stream=True, conf=0.3, imgsz=1920)

    # 결과에서 box들 저장
    for result in results:
        boxes = result.boxes

    # 'person'으로 인식된 객체 수 계산
    person_index = np.where(boxes.cls.cpu().numpy() == 0)[0]
    person_count = len(person_index)

    # 바운딩 박스 그리기 (이미지 저장을 위함)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        label = result.names[int(box.cls)]
        confidence = box.conf[0]
        
        # 사람으로 인식된 객체에만 바운딩 박스 추가
        if label == 'person':
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'{label} {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 이미지 저장
    output_path = os.path.join(current_dir, f'analyzed_image{camera_index+1}.jpg')
    cv2.imwrite(output_path, image)

    analyzed_data = {
        "camera_num": camera_index+1,
        "person_count": person_count,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return jsonify({"status": "success", "person_count": person_count, "camera_num": camera_index+1}), 200

@app.route('/data', methods=['GET'])
def get_data():
    global analyzed_data
    return jsonify(analyzed_data)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8000)
