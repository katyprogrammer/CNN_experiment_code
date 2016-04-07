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
    split = int(df.shape[0]*split_ratio)
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

def NN(epoch):
    net = NeuralNet(layers=[
        ('input', layers.DenseLayer), ('hidden', layers.DenseLayer), ('output', layers.DenseLayer)],
                    input_shape = (None, 1, 28, 28*2),
                    hidden_num_units = 100,
                    output_num_units = 4,
                    update = nesterov_momentum,
                    update_learning_rate = 1e-4,
                    update_momentum = 0.9,
                    max_epochs = epoch,
                    verbose = 1
    )
    return net
    

SPLIT_RATIO = 0.9
NUM = 5
A, B = [1,3], [2,4]
A_train, A_test = get_classes(A, SPLIT_RATIO, NUM)
B_train, B_test = get_classes(B, SPLIT_RATIO, NUM)
ORIGIN_HEADER = ['pixel{0}'.format(x) for x in range(28*28)]
ORIGIN_HEADER += ['label']
HEADER = ['{0}_{1}'.format(domain, column) for domain in ['A','B'] for column in ORIGIN_HEADER]
df_train = pd.DataFrame({x:[]} for x in HEADER)
for aidx, a in A_train.iterrows():
    for bidx, b in B_train.iterrows():
        tmp_dict = {}
        for x in ORIGIN_HEADER:
            tmp_dict.update({'A_{0}'.format(x): a[x]})
            tmp_dict.update({'B_{0}'.format(x): b[x]})
        df_train = df_train.append(tmp_dict, ignore_index=True)
df = None
for a in A:
    if df is None:
        df = df_train[df_train['A_label']==a]
    else:
        df = df.append(df_train[df_train['A_label']==a])
df.to_csv('train.csv')


# df_train = pd.read_csv('train.csv')
# label = []
# for idx, row in df_train.iterrows():
#     print row['A_label']
#     label += ['{0}{1}'.format(row['A_label'],row['B_label'])]
# print(label[:2])
# df_train['label'] = label
