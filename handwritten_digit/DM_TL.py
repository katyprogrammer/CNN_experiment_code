import pandas as pd
import numpy as np
import lasagne
import theano.tensor as T
import matplotlib.pyplot as plt
from lasagne.layers import InputLayer, DenseLayer
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from sklearn.metrics import accuracy_score
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


def multilabel_objective(predictions, targets):
    epsilon = np.float32(1e-6)
    one = np.float32(1.0)
    pred = T.clip(predictions, epsilon, one-epsilon)
    return -T.sum(targets*T.log(pred) + (one-targets)*T.log(one-pred), axis=1) # cross-entropy


def control_layer_num(n, l):
    tl = l
    for i in range(n):
        tl = DenseLayer(tl, num_units=HN, nonlinearity=None)
    return tl


def NN(epoch):
    l = InputLayer(name='input', shape=(None,1,28,28*2))
    l = control_layer_num(10, l)
    l = DenseLayer(l, num_units=10, nonlinearity=lasagne.nonlinearities.sigmoid)
    net = NeuralNet(l,
                    update = nesterov_momentum,
                    update_learning_rate = 1e-5,
                    update_momentum = 0.9,
                    max_epochs = epoch,
                    verbose = 1,
                    regression = True,
                    objective_loss_function = multilabel_objective
    )
    return net


def calc_shared(net):
    params = net.get_all_params_values()
    mid_layers = params.keys()[1:-1]
    shared = []
    last_layer = None
    for layer in mid_layers:
        cur_shared = []
        neuron_num = len(params[layer][1])
        x_num = len(params[layer][0])
        for neuron in range(neuron_num):
            abs_weight = np.array([abs(params[layer][0][i][neuron]) for i in range(x_num)])
            norm = sum(abs_weight)
            abs_weight /= norm
            if last_layer is None:
                last_layer = [1 for i in range(28*28)]
                last_layer += [0 for i in range(28*28)]
                last_layer = np.array(last_layer)
            cur_shared += [np.dot(abs_weight, last_layer)]
        shared += [sum(cur_shared)/neuron_num]
        last_layer = np.array(cur_shared)
    print(shared)
    plt.plot(shared)
    plt.show()

def gen_data(A, B, split_ratio, num, filename):
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
    df.to_csv(filename)

def run(filename):
    df_train = pd.read_csv(filename)
    label = []
    for idx, row in df_train.iterrows():
        l = np.zeros(10)
        l[int(row['A_label'])] = 1
        l[int(row['B_label'])] = 1
        label += [l]

    df_train['label'] = label
    df_train.to_csv('combined_{0}'.format(filename))
    HEADER = ['label']
    HEADER += ['A_pixel{0}'.format(x) for x in range(28*28)]
    HEADER += ['B_pixel{0}'.format(x) for x in range(28*28)]
    train = df_train[HEADER[1:]].values
    train = np.array(train).reshape((-1,1,28,28*2)).astype(np.uint8)
    label = np.array(label).astype(np.uint8)
    net = NN(100)
    net.fit(train, label)
    calc_shared(net)

SPLIT_RATIO = 0.9
NUM = 10
HN = 100

fname = 'low.csv'
A, B = [1,7], [0,9]
# gen_data(A,B,SPLIT_RATIO,NUM,fname)
run(fname)

fname = 'mid.csv'
A, B = [1,0], [7,9]
# gen_data(A,B,SPLIT_RATIO,NUM,fname)
run(fname)

fname = 'high.csv'
A, B = [1,7,0], [0,1,9]
# gen_data(A,B,SPLIT_RATIO,NUM,fname)
run(fname)