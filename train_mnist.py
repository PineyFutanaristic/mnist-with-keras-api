from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
import urllib.request

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255

inputs = tf.keras.layers.Input(shape=(28, 28, 1),  name='digits')
x = tf.keras.layers.Conv2D(16, 3, activation='relu', name='conv1')(inputs)
x = tf.keras.layers.Conv2D(16, 3, activation='relu', name='conv2')(x)
x = tf.keras.layers.MaxPool2D(2)(x)
x = tf.keras.layers.Conv2D(8, 3, activation='relu', name='conv3')(x)
x = tf.keras.layers.Conv2D(8, 3, activation='relu', name='conv4')(x)
x = tf.keras.layers.MaxPool2D(2)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(32, activation='relu', name='dense1')(x)
x = tf.keras.layers.Dense(32, activation='relu', name='dense2')(x)
x = tf.keras.layers.Dense(32, activation='relu', name='dense3')(x)
outputs = tf.keras.layers.Dense(10, activation='softmax', name='soft')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name='mnist')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=64, epochs=7, validation_split=0.2)
test_scores = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])

model.save("mnist_model.h5")

def show_img(img):
    plt.figure()
    plt.imshow(img, cmap=plt.cm.binary)
    plt.colorbar()
    plt.grid(False)
    plt.show()

def make_prediction(url):
    image = Image.open(urllib.request.urlopen(url))
    image = image.resize((28, 28)).convert('L')
    image = np.array(image).astype('float32') / 255.0
    image = np.vectorize(lambda x: 1 - x)(image)
    show_img(image)
    image = np.expand_dims(image, axis=2)
    image = np.array([image])
    prediction = model.predict(image)
    answer = np.argmax(prediction)
    # print(prediction)
    print("The model gives a prediction of {}".format(answer))

