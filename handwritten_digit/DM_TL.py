import pandas as pd
import numpy as np
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
import os

def read_and_split(filepath, split_ratio, num):
    df = pd.read_csv(filepath)
    df = df.iloc[:num/split_ratio]
    split = df.shape[0]*split_ratio
    train = df.iloc[:split]
    test = df.iloc[split:]
    return train, test

def get_classes(classes, split_ratio, num):
    train, test = None, None
    for digit in classes:
        tr, te = read_and_split('train/{0}.csv'.format(digit), split_ratio, num)
        if train is None:
            train, test = tr, te
        else:
            train = train.append(tr)
            test = test.append(te)
    return train, test

SPLIT_RATIO = 0.9
NUM = 500
A, B = [1,3], [2,4]
A_train, A_test = get_classes(A, SPLIT_RATIO, NUM)
B_train, B_test = get_classes(B, SPLIT_RATIO, NUM)
print A_train.shape
print B_train.shape