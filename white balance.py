import cv2
import numpy as np

def find_brightest_point(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(gray_image)
    return maxLoc

def adjust_white_balance(image, gamma=1.0):
    brightest_point = find_brightest_point(image)
    reference_white = np.array(image[brightest_point[1], brightest_point[0]], dtype=np.float32)
    image_float = np.array(image, dtype=np.float32)

    # Normalize the image using the reference white point
    image_normalized = np.divide(image_float, reference_white)
    image_normalized = np.clip(image_normalized, 0, 255)
    
    # Apply gamma correction
    image_gamma_corrected = np.power(image_normalized, gamma) * 255
    image_gamma_corrected = np.clip(image_gamma_corrected, 0, 255).astype(np.uint8)
    
    return image_gamma_corrected

# Load an image
input_image = cv2.imread("C:/Users/SAMSUNG/Desktop/input.jpg")

# Adjust the white balance
output_image = adjust_white_balance(input_image, gamma=1.8)

# Save the output image
cv2.imwrite("output.jpg", output_image)


