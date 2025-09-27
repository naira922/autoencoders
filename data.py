import os
import numpy as np
import pandas as pd
import kagglehub

def load_fashion_mnist():
    # Download dataset
    path = kagglehub.dataset_download("zalando-research/fashionmnist")
    print("Path to dataset files:", path)

    train_csv = os.path.join(path, "fashion-mnist_train.csv")
    test_csv = os.path.join(path, "fashion-mnist_test.csv")

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Normalize and reshape
    x_train = train_df.iloc[:, 1:].values.astype("float32") / 255.0
    x_test = test_df.iloc[:, 1:].values.astype("float32") / 255.0

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    print("Train:", x_train.shape, " Test:", x_test.shape)
    return x_train, x_test
