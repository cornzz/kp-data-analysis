import numpy as np
import pandas as pd
import keras
from keras.utils import plot_model, to_categorical
from keras.datasets import mnist
from keras.models import Sequential
from keras import Input
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout


# -------------- Load and reshape data --------------
# Load training and test data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# Reshape and normalize
X_train = np.expand_dims(X_train, -1) / 255
X_test = np.expand_dims(X_test, -1) / 255
# Convert labels to binary class vectors (One-hot encoding)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# -------------- Initial model --------------
# Build model
model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, kernel_size=3, padding='same', activation='relu'),
    Conv2D(64, kernel_size=3, padding='same', activation='relu'),
    MaxPool2D(pool_size=2),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
plot_model(model, to_file='model.png', show_shapes=True)

# Train model
model.fit(X_train, y_train, batch_size=32, epochs=12)

# Evaluate model
model.evaluate(X_test, y_test)
# 313/313 [==============================] - 8s 26ms/step - loss: 0.0628 - accuracy: 0.9899

# -------------- Improved model --------------
# This model has another max pooling layer in between the convolution layers,
# aswell as a dropout layer to prevent overfitting, providing better test accuracy
model2 = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, kernel_size=5, activation='relu'),
    MaxPool2D(pool_size=2),
    Conv2D(64, kernel_size=3, activation='relu'),
    MaxPool2D(pool_size=2),
    Flatten(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model2.fit(X_train, y_train, batch_size=32, epochs=12)

model2.evaluate(X_test, y_test)
# 313/313 [==============================] - 3s 9ms/step - loss: 0.0212 - accuracy: 0.9940
