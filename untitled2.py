import cv2
import numpy as np

def white_balance(img):
    result = cv2.xphoto.createSimpleWB().balanceWhite(img)
    return result

def skin_extract(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin = cv2.bitwise_and(img, img, mask=mask)
    return skin

def recommend_color(skin_bgr):
    skin_rgb = skin_bgr[::-1]
    r, g, b = skin_rgb

    if r > g and r > b:
        return "따뜻한 봄 컬러가 어울립니다."
    elif g > r and g > b:
        return "시원한 여름 컬러가 어울립니다."
    elif b > r and b > g:
        return "침착한 가을 컬러가 어울립니다."
    else:
        return "깨끗한 겨울 컬러가 어울립니다."

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    wb_frame = white_balance(frame)
    skin_frame = skin_extract(wb_frame)

    # 피부색 영역 추출하여 평균 피부색 계산
    skin_mask = skin_frame > 0
    skin_pixels = frame[skin_mask].mean(axis=0)
    personal_color = recommend_color(skin_pixels)

    cv2.imshow('Original', frame)
    cv2.imshow('White Balanced', wb_frame)
    cv2.imshow('Skin', skin_frame)
    print(personal_color)

    k = cv2.waitKey(1)
    if k == 27: # Esc 키를 누르면 종료합니다.
        break

cv2.release()
cv2.destroyAllWindows()


