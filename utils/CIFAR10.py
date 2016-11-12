"""
CIFAR10 data introduction: http://www.cs.toronto.edu/~kriz/cifar.html
"""

import os
import pickle
import numpy as np

CIFAR10_DATA_PATH = 'datasets/CIFAR10/cifar-10-batches-py'

def load_train_data():
    """
    Load training data of CIFAR10.
    There are 50000 32x32 images in 10 classes, with 5000 images per class.

    Returns
    -------
    X_train: A (50000, 3072) numpy array.
             Each row of the array stores a 32x32 colour image.
             The first 1024 entries contain the red channel values,
             the next 1024 the green, and the final 1024 the blue.
    y_train: A (50000,) numpy array.
    """
    data_path_base = os.path.join(CIFAR10_DATA_PATH, 'data_batch_')
    Xs = []
    ys = []
    for i in range(1, 6):
        data_path = data_path_base + str(i)
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            X = data['data']
            y = np.array(data['labels'])
            Xs.append(X)
            ys.append(y)
    X_train = np.concatenate(Xs)
    y_train = np.concatenate(ys)
    return X_train, y_train

def load_test_data():
    """
    Load test data of CIFAR10.
    There are 10000 32x32 images in 10 classes, with 1000 images per class.

    Returns
    -------
    X_test: A (10000, 3072) numpy array.
            Each row of the array stores a 32x32 colour image.
            The first 1024 entries contain the red channel values,
            the next 1024 the green, and the final 1024 the blue.
    y_test: A (10000,) numpy array.
    """
    data_path = os.path.join(CIFAR10_DATA_PATH, 'test_batch')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        X_test = data['data']
        y_test = np.array(data['labels'])
    return X_test, y_test

def load_label_names():
    """
    Load the names of the 10 classes labels.

    Return a 10-element list.
    """
    batches_meta_path = os.path.join(CIFAR10_DATA_PATH, 'batches.meta')
    with open(batches_meta_path, 'rb') as f:
        batches_meta = pickle.load(f)
    return batches_meta['label_names']
