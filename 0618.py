import cv2
import numpy as np

#Load the cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#Function for finding the average color in an image region
def average_color(image_region):
    return np.mean(image_region, axis=(0, 1)).astype(int)
#Function for determining the personal color based on skin color
def personal_color_recommendation(skin_color):
    h, s, v = cv2.cvtColor(
        np.uint8([[skin_color]]), cv2.COLOR_BGR2HSV)[0][0]
    if h < 15 or h > 165:
        personal_color = 'warm'
    elif 15 <= h < 45:
        personal_color = 'spring'
    elif 45 <= h < 85:
        personal_color = 'cool'
    else:
        personal_color = 'autumn'

    return personal_color

#Start the camera
cap = cv2.VideoCapture(0) 

while True:
# Capture frame-by-frame
    ret, frame = cap.read()
# Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=7)

# Find the largest face (in case multiple faces are detected)
    if len(faces) > 0:
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        faces = [largest_face]

# Process the detected face(s) and extract skin color
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_color = frame[y:y + h, x:x + w]
        
        # Extract skin color using the white balance technique
        skin_color = average_color(roi_color)
        
        # Determine the personal color that suits the person
        personal_color = personal_color_recommendation(skin_color)
        
        # Write the personal color on the frame
        cv2.putText(frame, f"Personal Color: {personal_color}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display the resulting frame
    cv2.imshow('Personal Color Detection', frame)

# Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#Release the camera and destroy all windows

cap.release()
cv2.destroyAllWindows()