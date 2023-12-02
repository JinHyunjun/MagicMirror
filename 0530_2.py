import cv2
import numpy as np

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

    if r > 210 and g > 210 and b > 170:
        return "spring"
    elif r < 210 and g < 210 and b > 170:
        return "summer"
    elif r < 210 and g < 210 and b < 170:
        return "fall"
    else:
        return "winter"
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    white_balanced_frame = apply_grayworld_white_balance(frame)
    skin_image = detect_skin(white_balanced_frame)
    average_color = average_skin_color(skin_image)
    personal_color = recommend_personal_color(average_color)

    cv2.putText(skin_image, personal_color, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Original', frame)
    cv2.imshow('White Balanced', white_balanced_frame)
    cv2.imshow('Skin Detected', skin_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

