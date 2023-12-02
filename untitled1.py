import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('mnist_final.csv').to_numpy()
labels = data[:, 0]
images = data[:, 1:].reshape(-1, 28, 28)

digits = [5, 6, 8, 9]
indices = [np.where(labels == d)[0][99] for d in digits]

for i, idx in enumerate(indices):
    plt.subplot(1, 4, i + 1)
    plt.imshow(images[idx], cmap='gray')
    plt.title(f'Digit: {labels[idx]}')
plt.show()

def diagonal_elements(array):
    return np.diag(array)

def probability_density(array):
    return array / array.sum()

def expected_value(array):
    return np.average(array)

def variance(array):
    return np.var(array)

def zero_count(array):
    return (array == 0).sum()

features = []
for idx in indices:
    image = images[idx]
    diagonal = diagonal_elements(image)

    # 1번
    feature1 = expected_value(diagonal)

    # 2번
    feature2 = variance(diagonal)

    # 3번
    feature3 = zero_count(diagonal)

    horizontal_projection = image.sum(axis=1)
    pdf_horizontal = probability_density(horizontal_projection)

    # 4번
    feature4 = expected_value(pdf_horizontal)

    # 5번
    feature5 = variance(pdf_horizontal)

    features.append([feature1, feature2, feature3, feature4, feature5])

features_matrix = np.array(features)
print(features_matrix)

def train_test_split(X, y, train_size=0.7, random_state=None):
    if random_state:
        np.random.seed(random_state)

    indices = np.random.permutation(len(X))
    train_indices = indices[:int(train_size * len(X))]
    test_indices = indices[int(train_size * len(X)):]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test

X = data[:, 1:]
y = data[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.8, random_state=42)

