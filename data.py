import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.datasets import fashion_mnist

def load_fashion_mnist():
    (x_train, _), (x_test, _) = fashion_mnist.load_data()

    # Normalize to [0,1] and add channel dimension
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)  # (N,28,28,1)
    x_test = np.expand_dims(x_test, -1)

    return x_train, x_test 