import cv2
import numpy as np

# Load the pre-trained face detector and eye detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Define a dictionary of skin tone colors
skin_colors = {
    "light": (255, 218, 170),
    "medium": (205, 133, 63),
    "dark": (139, 69, 19)
}

# Define a dictionary of pupil colors
pupil_colors = {
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "brown": (0, 0, 255)
}

# Capture video from the default camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # For each face detected, draw a rectangle around it and suggest a clothing color
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract the ROI (region of interest) corresponding to the face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes in the ROI
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)

        # If two eyes are detected, determine the pupil color and suggest a clothing color
        if len(eyes) == 2:
            # For each eye detected, draw a rectangle around it and determine the pupil color
            for (ex, ey, ew, eh) in eyes:
                # Draw a rectangle around the eye
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

                # Extract the ROI (region of interest) corresponding to the eye
                eye_roi_color = roi_color[ey:ey + eh, ex:ex + ew]

                # Calculate the average color of the ROI to determine the pupil color
                avg_color_per_row = np.average(eye_roi_color, axis=0)
                avg_color = np.average(avg_color_per_row, axis=0)
                bgr_color = np.uint8([[avg_color]])
                hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)

                # Determine the pupil color based on the HSV color space
                if hsv_color[0][0][0] >= 90 and hsv_color[0][0][0] <= 130:
                    pupil_color_name = "green"
                elif hsv_color[0][0][0] >= 5 and hsv_color[0][0][0] <= 30:
                    pupil_color_name = "brown"
                else:
                    pupil_color_name = "blue"

            # Determine the skin tone color based on the average color of the face ROI
            avg_color_per_row = np.average(roi_color, axis=0)
            avg_color = np.average(avg_color_per_row, axis=0)
            bgr_color = np.uint8([[avg_color]])
            hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)

            # Determine the clothing color based on the pupil and skin tone colors
            if pupil_color_name == "blue":
                clothing_color_name = "light"
            elif pupil_color_name == "green":
                if hsv_color[0][0][1] >= 100:
                    clothing_color_name = "medium"
                else:
                    clothing_color_name = "light"
            else:  # pupil_color_name == "brown"
                if hsv_color[0][0][2] >= 100:
                    clothing_color_name = "dark"
                else:
                    clothing_color_name = "medium"

            # Draw a rectangle around the suggested clothing color
            cv2.rectangle(frame, (x + w + 10, y), (x + w + 100, y + 30), skin_colors[clothing_color_name], -1)
            cv2.putText(frame, clothing_color_name.capitalize() + " clothes", (x + w + 15, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
