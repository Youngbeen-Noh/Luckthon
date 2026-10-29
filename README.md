# 실시간 혼잡도 지도 서비스 (럭키톤 2024)

카메라로 찍은 사진을 YOLO11로 분석해 사람 수를 세고, 건물 평면도 위에 구역별 혼잡도를 실시간으로 보여주는 서비스입니다. 인천대학교 정보기술대학 학생회 해커톤(2024.10.27 – 11.02)에서 만들었습니다.

## 구조

```
Raspberry Pi (Picamera2)  ──POST /upload──▶  Flask 서버 (YOLO11 사람 검출 · 인원 산출)
                                                  │
                         ┌────────────────────────┴───────────────────────┐
                    관리자 페이지 /manager                           사용자 페이지 /
           평면도 업로드 · 카메라 위치 지정 · 혼잡 기준 설정        혼잡도 지도 조회 (5초 갱신)
```

- 관리자가 정한 인원 기준에 따라 카메라별 혼잡도를 4단계 아이콘 색(빨강·주황·노랑·초록)으로 표시하고, 아이콘을 누르면 검출 결과 이미지를 보여줍니다.
- 무거운 yolo11x 대신 경량 모델(yolo11n)과 높은 입력 해상도를 조합해 응답 속도와 원거리 인식을 맞췄습니다.

## 주요 API

| Method | Path | 설명 |
|---|---|---|
| POST | `/upload` | 카메라 이미지 업로드 → 인원 산출 |
| GET | `/get_camera_data` | 카메라별 인원·상태 조회 |
| GET | `/image/<camera_name>` | 검출 결과 이미지 |
| GET / POST | `/get_page_data`, `/save_page_data` | 평면도·카메라 배치 불러오기/저장 |
| POST | `/reset_data` | 데이터 초기화 |

## 실행

```bash
pip install flask ultralytics opencv-python
python server.py                # 0.0.0.0:8000
# 라즈베리파이에서 (raspberrypi/camera.py의 서버 주소를 수정한 뒤)
python raspberrypi/camera.py
```

## 역할

- **노영빈**: Flask 서버(업로드 API, YOLO11 인원 산출, 카메라 상태 관리), 관리자/사용자 페이지 연동
- 팀원: 초기 웹 페이지와 혼잡도 아이콘, 카메라 모듈 연동

## 기술

Python · Flask · Ultralytics YOLO11 · OpenCV · Raspberry Pi(Picamera2) · HTML/JS
