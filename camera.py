from picamera2 import Picamera2
import time
import requests

# 서버 URL과 이미지 파일 경로 설정
# url = 'http://wcl.inu.ac.kr:8000/upload'  # 서버 URL을 실제 URL로 변경
url = 'http://192.168.107.91:8000/upload'
file_path = '/home/pi/post-image/images/Camera_1.jpg'

picam2 = Picamera2()
config = picam2.create_still_configuration()
picam2.configure(config)
picam2.start()

# time.sleep(2)

picam2.capture_file(file_path)
picam2.stop()

# 이미지 파일을 열고 서버에 POST 요청 전송
with open(file_path, 'rb') as img_file:
    files = {'file': img_file}
    response = requests.post(url, files=files)

# 응답 확인
if response.status_code == 200:
    print("이미지 업로드 성공!")
else:
    print(f"업로드 실패, 상태 코드: {response.status_code}")
