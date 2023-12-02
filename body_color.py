import cv2

# Load the Haar Cascade
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_fullbody.xml')

def detect_body(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bodies = body_cascade.detectMultiScale(gray, 1.1, 5)
    return bodies

def recommend_clothes(body_shape):
    clothes_recommendation = ""

    if body_shape == "triangle":
        clothes_recommendation = "와이드레그 팬츠와 라지사이즈의 상의를 추천합니다."
    elif body_shape == "inverted_triangle":
        clothes_recommendation = "사이즈가 큰 하의와 벨트를 활용한 스타일을 추천합니다."
    elif body_shape == "rectangle":
        clothes_recommendation = "벨트를 사용한 드레스와 하이웨이스트 팬츠를 추천합니다."
    elif body_shape == "hourglass":
        clothes_recommendation = "타이트한 드레스와 딱 맞는 상의와 하의를 추천합니다."
    elif body_shape == "apple":
        clothes_recommendation = "루즈한 상의와 야구모자를 추천합니다."
    elif body_shape == "pear":
        clothes_recommendation = "발레가 딱 맞는 데님 자켓을 추천합니다."
    else:
        clothes_recommendation = "유효한 체형을 올바르게 입력해주세요."

    return clothes_recommendation

# 사용 예제
body_shape = "hourglass"
recommendation = recommend_clothes(body_shape)
print(f"{body_shape} 체형에 대한 추천 옷: {recommendation}")

# Capture video from camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect bodies in the frame
    bodies = detect_body(frame)

    # Draw rectangles around the bodies
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Recommend clothes based on body_size
        body_size = (w, h)
        recommend_clothes(body_size)

    # Show the frame
    cv2.imshow('Body Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
