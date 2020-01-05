from __future__ import absolute_import, division, print_function, unicode_literals
from PIL import Image

import sys
import tensorflow_hub as hub
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import urllib.request

model = tf.keras.models.load_model("mnist_model.h5", custom_objects={'KerasLayer': hub.KerasLayer})
model.summary()

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
    print(prediction)
    print("The model gives a prediction of {}".format(answer))

make_prediction("https://data.ac-illust.com/data/thumbnails/d5/d54404572b4528dfc071e79b57413724_t.jpeg")

sys.exit()
