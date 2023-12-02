import cv2
import numpy as np
import time

def detect_faces(img, cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def get_skin_tone(face, img):
    x, y, w, h = face
    skin_roi = img[y:y + h, x:x + w]
    avg_skin_tone = np.mean(skin_roi, axis=(0, 1))
    return avg_skin_tone

def recommend_personal_color(avg_skin_tone):
    r, g, b = avg_skin_tone
    if r > 165 and g > 145 and b < 130:
        return "Warm-toned colors are recommended."
    elif r < 150 and g < 130 and b > 120:
        return "Bold, cool-toned colors are recommended."
    else:
        return "Soft pastel tones are recommended."

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

last_output_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to load camera. Please try again.")
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
