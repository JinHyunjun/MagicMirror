#pip3 install picamera

import cv2
import numpy as np
import picamera
from picamera.array import PiRGBArray

def apply_grayworld_white_balance(image):
    image_float = np.float32(image)
    channel_avg = np.mean(image_float, axis=(0, 1))

    # 채널별로 픽셀 값을 평균낸 뒤 평균 값을 128로 만들어 각 채널의 밝기를 조절합니다.
    scaling_factor = 128.0 / channel_avg

    white_balanced_image = np.multiply(image_float, scaling_factor)
    white_balanced_image = np.clip(white_balanced_image, 0, 255) # 0~255 범위로 제한
    white_balanced_image = np.uint8(white_balanced_image)
    
    return white_balanced_image


def detect_skin(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_skin_hsv = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin_hsv = np.array([20, 255, 255], dtype=np.uint8)

    skin_mask = cv2.inRange(hsv_image, lower_skin_hsv, upper_skin_hsv)
    result = cv2.bitwise_and(image, image, mask=skin_mask)

    return result

def average_skin_color(skin_image):
    num_skin_pixels = np.sum(skin_image > 0) // 3
    average_color = np.sum(skin_image, axis=(0, 1)) / num_skin_pixels

    return average_color.astype(np.uint8)

def recommend_personal_color(average_color):
    r, g, b = average_color

    spring_colors = ["pink", "lavender", "coral", "mint", "loveblossom"]
    summer_colors = ["skyblue", "aqua", "water", "melon", "ocean"]
    fall_colors = ["orange", "mustard", "tomato", "maple", "pumpkin"]
    winter_colors = ["plum", "emerald", "berry", "white", "winter sky"]

    if r > 210 and g > 210 and b > 170:
        return "spring", spring_colors
    elif r < 210 and g < 210 and b > 170:
        return "summer", summer_colors
    elif r < 210 and g < 210 and b < 170:
        return "fall", fall_colors
    else:
        return "winter", winter_colors

cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 카메라 설정
width = 1280
height = 960
with picamera.PiCamera() as camera:
    camera.resolution = (width, height)
    camera.framerate = 32
    rawCapture = PiRGBArray(camera, size=(width, height))
    
    # 프레임별로 캡처
    for capture in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        frame = capture.array
        white_balanced_frame = apply_grayworld_white_balance(frame)

        faces = cascade.detectMultiScale(white_balanced_frame, scaleFactor=1.3, minNeighbors=7, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(white_balanced_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = white_balanced_frame[y:y+h, x:x+w]
            skin_image = detect_skin(face)
            average_color = average_skin_color(skin_image)
            personal_color, color_codes = recommend_personal_color(average_color)
        
            cv2.putText(white_balanced_frame, f"{personal_color}: {', '.join(color_codes)}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('White Balanced', white_balanced_frame)

        key = cv2.waitKey(1) & 0xFF
        
        # 현재 프레임의 버퍼를 비워준다.
        rawCapture.truncate(0)

        if key == ord('q'):
            break

cv2.destroyAllWindows()
