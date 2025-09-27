import numpy as np
from tensorflow.keras.datasets import fashion_mnist

def load_fashion_mnist():
    # Load dataset
    (x_train, _), (x_test, _) = fashion_mnist.load_data()

    # Normalize to [0,1] and reshape for CNN input
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add channel dimension (needed for Conv layers)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    return x_train, x_test
