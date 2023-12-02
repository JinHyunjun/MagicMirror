import pandas as pd
import numpy as np
image = pd.read_csv(r"C:/Users/SAMSUNG/Desktop/csv_image.csv", encoding="cp949") / 255
label = pd.read_csv(r"C:/Users/SAMSUNG/Desktop/csv_label.csv", encoding="cp949")
image_array = np.array(image, dtype='f')
label_array = np.array(label, dtype='f')

import matplotlib.pyplot as plt

# 이미지 데이터를 3차원 배열로 변경
num_images = len(image_array)
image_width = 32
image_height = 32
num_channels = 3

image_3D_array = image_array.reshape(num_images, image_height, image_width, num_channels)

# 100번째 이미지를 출력
plt.imshow(image_3D_array[99])
plt.show()