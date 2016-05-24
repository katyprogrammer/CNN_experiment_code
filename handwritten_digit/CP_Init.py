import pandas as pd
import numpy as np
import lasagne
from nolearn.lasagne import TrainSplit
import theano.tensor as T
import matplotlib.pyplot as plt
from lasagne.layers import InputLayer, DenseLayer
from lasagne.updates import nesterov_momentum
from NeuralNet import NeuralNet
import cPickle

# data processing
def read_and_split(filepath):
    df = pd.read_csv(filepath)
    df = df.iloc[:NUM/SPLIT_RATIO]
    split = int(df.shape[0]*SPLIT_RATIO)
    train = df.iloc[:split]
    test = df.iloc[split:]
    return train, test

def get_classes(classes):
    train, test = None, None
    for digit in classes:
        tr, te = read_and_split('train/{0}.csv'.format(digit))
        if train is None:
            train, test = tr, te
        else:
            train = train.append(tr)
            test = test.append(te)
    return train, test

def gen_data(A, B):
    A_train, A_test = get_classes(A)
    B_train, B_test = get_classes(B)
    A_train.to_csv('{0}_A_train.csv'.format(RUN_NAME))
    B_train.to_csv('{0}_B_train.csv'.format(RUN_NAME))
    A_test.to_csv('{0}_A_test.csv'.format(RUN_NAME))
    B_test.to_csv('{0}_B_test.csv'.format(RUN_NAME))

def read_data(filename):
    df = pd.read_csv(filename)
    header = ['pixel{0}'.format(x) for x in range(28*28)]
    data = np.array(df[header].values).reshape((-1,1,28,28)).astype(np.uint8)
    label = []
    for idx, row in df.iterrows():
        l = np.zeros(10)
        l[int(row['label'])] = 1
        label += [l]
    label = np.array(label).astype(np.uint8)
    return data, label
    

# training
def objective(predictions, targets):
    one = np.float32(1.0)
    return -T.sum(targets*T.log(predictions) + (one-targets)*T.log(one-predictions), axis=1)

def control_layer_num(n, l):
    tl = l
    for i in range(n):
        tl = DenseLayer(tl, num_units=HN, nonlinearity=None)
    return tl

def NN(epoch, custom_regularizor=None):
    l = InputLayer(name='input', shape=(None,1,28,28))
    l = control_layer_num(LN, l)
    l = DenseLayer(l, name='output', num_units=10, nonlinearity=lasagne.nonlinearities.sigmoid)
    net = NeuralNet(l,
                    update = nesterov_momentum,
                    update_learning_rate = 1e-5,
                    update_momentum = 0.9,
                    max_epochs = epoch,
                    verbose = 1,
                    train_split = TrainSplit(eval_size=0.2),
                    regression = True,
                    objective_loss_function = objective,
                    custom_regularizor = custom_regularizor,
    )
    return net

def select_max(x):
    A = np.argmax(x)
    return np.array([1 if i==A else 0 for i in range(10)])

def save_params(A_OR_B, net):
    cPickle.dump(net.get_all_params_values(), open('{0}_{1}_net.pkl'.format(RUN_NAME, A_OR_B), 'w+'))

def run(A_OR_B, CP_Init=False):
    train, train_label = read_data('{0}_{1}_{2}.csv'.format(RUN_NAME, A_OR_B, 'train'))
    test, test_label = read_data('{0}_{1}_{2}.csv'.format(RUN_NAME, A_OR_B, 'test'))
    while True:
        net = NN(EPOCH)
        net.fit(train, train_label)
        pred = net.predict(test)
        n = len(pred)
        acc = 0
        for i in range(n):
            p = select_max(pred[i])
            compare = [0 if p[x]==test_label[i][x] else 1 for x in range(10)]
            acc = acc+1 if sum(compare) == 0 else acc
        accuracy = float(acc)/n
        print('accuracy={0}'.format(accuracy))
        if accuracy > ACC:
            break
    save_params(A_OR_B, net)
    return net

# train vs test
SPLIT_RATIO = 0.9
NUM = 1000
# neural configuration
LN = 10
HN = 50
ACC = 0.9
EPOCH = 50

RUN_NAME = '1st'
A, B = [1,7,4], [0,9,6]
gen_data(A, B)
net1 = run('A')
# net2 = run('B')