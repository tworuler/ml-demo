import os
import cPickle as pickle
import numpy as np

CIFAR10_DATA_PATH = 'datasets/CIFAR10/cifar-10-batches-py'

def load_train_data():
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
    data_path = os.path.join(CIFAR10_DATA_PATH, 'test_batch')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        X_test = data['data']
        y_test = data['labels']
    return X_test, y_test

def load_label_names():
    batches_meta_path = os.path.join(CIFAR10_DATA_PATH, 'batches.meta')
    with open(batches_meta_path, 'rb') as f:
        batches_meta = pickle.load(f)
    return batches_meta['label_names']
