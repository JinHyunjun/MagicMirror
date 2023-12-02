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

# 카메라를 사용하여 이미지를 캡처하기
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read() # 프레임 받아오기
    white_balanced_frame = apply_grayworld_white_balance(frame) # 화이트 밸런스 적용
    
    cv2.imshow('Original', frame)
    cv2.imshow('White Balanced', white_balanced_frame)

    # 종료할 경우
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
