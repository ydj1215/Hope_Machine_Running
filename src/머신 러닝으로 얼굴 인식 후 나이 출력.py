import cv2
import numpy as np

# 얼굴 인식을 위한 캐스케이드 분류기 로드
face_cascade = cv2.CascadeClassifier('../opencv/haarcascade_frontalface_default.xml')

# 안면 나이 추정 및 성별 추정을 위한 딥러닝 모델 로드
age_model = cv2.dnn.readNetFromCaffe('../opencv/age_deploy.prototxt', '../opencv/age_net.caffemodel')
gender_model = cv2.dnn.readNetFromCaffe('../opencv/gender_deploy.prototxt', '../opencv/gender_net.caffemodel')

# 연령 및 성별 리스트 정의
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']
age_mid = [1, 5, 10, 17.5, 28.5, 40.5, 50.5, 80]

# 이미지 파일 읽기
img = cv2.imread('../data/face.JPG')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 얼굴 검출
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# 이미지에 얼굴 위치를 표시합니다.
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# 수정된 이미지를 표시합니다.
cv2.imshow('Face Detection Check', img)

# 각 얼굴에 대한 처리
for (x, y, w, h) in faces:
    face_img = img[y:y + h, x:x + w]
    blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

    # 나이 추정
    age_model.setInput(blob)
    age_preds = age_model.forward()
    age = age_list[age_preds[0].argmax()]
    age_est = round(np.sum(age_mid * age_preds[0]))  # 나이 추정 결과를 계산하고 반올림

    # 성별 추정
    gender_model.setInput(blob)
    gender_preds = gender_model.forward()
    gender = gender_list[gender_preds[0].argmax()]

    # 결과 출력
    label = f'{gender}, {str(age_est)}, {age}'
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# 이미지 표시
cv2.imshow('Gender and Age Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
