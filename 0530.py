import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from tensorflow.keras.models import Model
import h5py

def load_dataset(filepath):
    with h5py.File(filepath, "r") as f:
        train_data = np.array(f["train_data"])
        train_targets = np.array(f["train_targets"])
        validation_data = np.array(f["validation_data"])
        validation_targets = np.array(f["validation_targets"])

    return (train_data, train_targets), (validation_data, validation_targets)

(data_train, targets_train), (data_val, targets_val) = load_dataset("prepared_white_balanced_data.h5")

# 데이터 정규화
X_train = data_train.astype('float32') / 255.0
Y_train = targets_train.astype('float32') / 255.0
X_val = data_val.astype('float32') / 255.0
Y_val = targets_val.astype('float32') / 255.0


def create_white_balance_model():
    input_layer = Input(shape=(128, 128, 3))
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    pool1 = MaxPooling2D((2, 2), padding='same')(conv1)

    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D((2, 2), padding='same')(conv2)

    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool2)

    upsample1 = UpSampling2D((2, 2))(conv3)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(upsample1)

    upsample2 = UpSampling2D((2, 2))(conv4)
    output_layer = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(upsample2)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    return model

model = create_white_balance_model()
model.summary()

history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=50, batch_size=16)
import cv2
import numpy as np

def apply_white_balance(image, model):
    resized_image = cv2.resize(image, (128, 128))   # 모델에 적합한 이미지 크기로 조정
    normalized_image = resized_image / 255.0        # 픽셀 값을 0~1 사이의 값으로 정규화
    predicted_image = model.predict(np.expand_dims(normalized_image, axis=0))
    white_balanced_image = np.clip(predicted_image, 0, 1) * 255  # 화이트 밸런스가 적용된 이미지로 변환
    return white_balanced_image

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    white_balanced_frame = apply_white_balance(frame, model)
    
    cv2.imshow('Original', frame)
    cv2.imshow('White Balanced', white_balanced_frame.astype('uint8'))

    # 종료할 경우
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
