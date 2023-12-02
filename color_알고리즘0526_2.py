import cv2
import numpy as np
import time

def find_brightest_point(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(gray_image)
    return maxLoc

def adjust_white_balance(image, gamma=1.0):
    brightest_point = find_brightest_point(image)
    reference_white = np.array(image[brightest_point[1], brightest_point[0]], dtype=np.float32)
    image_float = np.array(image, dtype=np.float32)

    image_normalized = np.divide(image_float, reference_white)
    image_normalized = np.clip(image_normalized, 0, 255)

    image_gamma_corrected = np.power(image_normalized, gamma) * 255
    image_gamma_corrected = np.clip(image_gamma_corrected, 0, 255).astype(np.uint8)

    return image_gamma_corrected

def classify_personal_color(skin_color):
    avg_intensity = np.mean(skin_color)

    if avg_intensity < 51:
        return 'Cool and Deep'
    elif avg_intensity < 102:
        return 'Warm and Light'
    elif avg_intensity < 153:
        return 'Cool and Light'
    elif avg_intensity < 204:
        return 'Warm and Deep'
    else:
        return 'Neutral'

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_skin_tone(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, (0, 20, 70), (20, 255, 255))
    skin_tone = cv2.bitwise_and(image, image, mask=mask)
    skin_tone_avg = cv2.mean(skin_tone, mask)
    return skin_tone_avg

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW를 사용하여 오류 문제를 해결합니다.
prev_recommendation_time = time.time()

while True:
    ret, img = cap.read()
    if not ret:
        print("카메라 동작에 문제가 발생했습니다.")
        break

    wb_img = adjust_white_balance(img, gamma=1.8)
    gray = cv2.cvtColor(wb_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    skin_tone = None

    for (x, y, w, h) in faces:
        cv2.rectangle(wb_img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 얼굴 영역 표시
        face_area = wb_img[y:y + h, x:x + w]
        skin_tone = get_skin_tone(face_area)

        if time.time() - prev_recommendation_time > 3:
            personal_color = classify_personal_color(skin_tone)
            print("Recommended personal color:", personal_color)
            prev_recommendation_time = time.time()

    cv2.imshow('adjusted image', wb_img)  # 밸런스 조정된 이미지를 보여줍니다

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
