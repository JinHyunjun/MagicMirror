import cv2
import numpy as np
import time

def detect_faces(img, cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def get_skin_tone(face, img):
    x, y, w, h = face
    face_roi = img[y:y + h, x:x + w]
    hsv_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)

    border_mask = cv2.rectangle(np.zeros((h, w), dtype=np.uint8), (0, 0), (w - 1, h - 1), 255, 3)
    border_pixels = hsv_face[np.where(border_mask == 255)]
    avg_hue = np.average(border_pixels[:, 0])

    lower_skin = np.array([avg_hue - 10, 20, 70])
    upper_skin = np.array([avg_hue + 10, 255, 255])

    skin_mask = cv2.inRange(hsv_face, lower_skin, upper_skin)
    skin_tone = cv2.bitwise_and(hsv_face, hsv_face, mask=skin_mask)
    average_skin_tone = np.average(skin_tone[np.where(skin_mask != 0)])

    return average_skin_tone

def recommend_personal_color(skin_tone):
    if skin_tone < 45:
        return "따뜻한 계열의 색상을 추천합니다."
    elif skin_tone < 90:
        return "부드러운 파스텔 톤의 색상을 추천합니다."
    else:
        return "진한, 시원한 계열의 색상을 추천합니다."

def main():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    last_output_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("카메라를 불러오지 못했습니다. 다시 시도하세요.")
            break

        faces = detect_faces(frame, face_cascade)

        for face in faces:
            x, y, w, h = face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if len(faces) > 0 and time.time() - last_output_time > 3:
            skin_tone = get_skin_tone(faces[0], frame)
            recommended_color = recommend_personal_color(skin_tone)
            print(recommended_color)
            last_output_time = time.time()

        cv2.imshow('Real-time Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


