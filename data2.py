import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import kagglehub  # Assuming you have kagglehub installed

# -----------------------------
# Load Fashion-MNIST CSV Dataset
# -----------------------------
path = kagglehub.dataset_download("zalando-research/fashionmnist")
print("Path to dataset files:", path)

train_csv = os.path.join(path, "fashion-mnist_train.csv")
test_csv = os.path.join(path, "fashion-mnist_test.csv")

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

y_train = train_df.iloc[:, 0].values
x_train = train_df.iloc[:, 1:].values
y_test = test_df.iloc[:, 0].values
x_test = test_df.iloc[:, 1:].values

# Reshape images to (28,28,1) and normalize
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

print("Train shape:", x_train.shape, " Test shape:", x_test.shape)

# Fashion-MNIST class labels
fashion_labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# -----------------------------
# Sampling Layer
# -----------------------------
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon