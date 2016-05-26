import pandas as pd
import numpy as np
import lasagne
from nolearn.lasagne import TrainSplit
import theano.tensor as T
import matplotlib.pyplot as plt
from lasagne.layers import InputLayer, DenseLayer
from lasagne.updates import nesterov_momentum
from NeuralNet import NeuralNet
import os
from os.path import join
import time
import cPickle

LOG = ""

# data processing
def read_and_split(filepath):
    df = pd.read_csv(filepath)
    if NUM is not None:
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

    global LOG
    LOG += "\ttrain\ttest\nA\t{0}\t{1}\nB\t{2}\t{3}\n".format(A_train.shape[0], A_test.shape[0], B_train.shape[0], B_test.shape[0])
    A_train.to_csv(join(RUN_NAME, '{0}_A_train.csv'.format(RUN_NAME)))
    B_train.to_csv(join(RUN_NAME, '{0}_B_train.csv'.format(RUN_NAME)))
    A_test.to_csv(join(RUN_NAME, '{0}_A_test.csv'.format(RUN_NAME)))
    B_test.to_csv(join(RUN_NAME, '{0}_B_test.csv'.format(RUN_NAME)))

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
    l = DenseLayer(l, name='input_1', num_units=HN, nonlinearity=None)
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
    net.save_params_to(join(RUN_NAME, '{0}_{1}_net.pkl'.format(RUN_NAME, A_OR_B)))

def run(A_OR_B, CP_R=None, LNum=None):
    global LOG
    train, train_label = read_data(join(RUN_NAME, '{0}_{1}_{2}.csv'.format(RUN_NAME, A_OR_B, 'train')))
    test, test_label = read_data(join(RUN_NAME, '{0}_{1}_{2}.csv'.format(RUN_NAME, A_OR_B, 'test')))
    while True:
        net = NN(EPOCH)
        # load trained parameters
        if CP_R is not None:
            # measure CP approximate time
            st = time.time()
            if LNum is None:
                global LN
                LNum = LN
            net.load_CP_approx_params_from(join(RUN_NAME, '{0}_{1}_net.pkl'.format(RUN_NAME, 'A')), HN, LNum, CP_R=CP_R)
            ed = time.time()
            LOG += "CP_approximate_exetime: {0}s\n".format(ed-st)
        # measure fitting time
        st = time.time()
        net.fit(train, train_label)
        ed = time.time()
        LOG += "training_time: {0}s\n".format(ed-st)
        pred = net.predict(test)
        n = len(pred)
        acc = 0
        for i in range(n):
            p = select_max(pred[i])
            compare = [0 if p[x]==test_label[i][x] else 1 for x in range(10)]
            acc = acc+1 if sum(compare) == 0 else acc
        accuracy = float(acc)/n
        LOG += "[{0}_{2}] acc = {1}\n".format(A_OR_B, accuracy, EXP_NAME)
        print('accuracy={0}'.format(accuracy))
        if accuracy > ACC:
            break
    if CP_R is None:
        save_params(A_OR_B, net)
        cPickle.dump(net.train_history_, open(join(RUN_NAME, '{0}_{1}.pkl'.format(A_OR_B, EXP_NAME)), 'w+'))
    else:
        save_params('{0}_R{1}'.format(A_OR_B, CP_R), net)
        cPickle.dump(net.train_history_, open(join(RUN_NAME, '{0}_{1}.pkl'.format(A_OR_B, EXP_NAME)), 'w+'))
    return net

# train vs test
SPLIT_RATIO = 6.0/7
NUM = None
# neural configuration
LN = 10 # layer number
HN = 50 # hidden unit per layer
ACC = 0.1
EPOCH = 35

RUN_NAME = 'MLP_{0}LN_{1}HN'.format(LN,HN)
if not os.path.exists(RUN_NAME):
    os.makedirs(RUN_NAME)


# logging
f = open(join(RUN_NAME, 'log_{0}.txt'.format(RUN_NAME)), 'a+')
A, B = [1,7,4,5,8], [2,3,6,0,9] # AB
# A, B = [2,3,6,0,9], [1,7,4,5,8] # BA
gen_data(A, B)
LOG += '---' * 9 + '\n'
EXP_NAME = 'Baseline'
net1 = run('A')
LOG += '---' * 9 + '\n'
f.write(LOG)
LOG = ""
EXP_NAME = 'Baseline'
net1 = run('B')
LOG += '---' * 9 + '\n'
f.write(LOG)

LOG = ""
for i in range(1,3):
    for j in range(1,LN):
        EXP_NAME = '{0}_rank1_{1}_Layer'.format(i, j)
        # transfer R low-rank approximation
        net2 = run('B', CP_R=i, LNum=j)
        LOG += '---' * 9 + '\n'
        f.write(LOG)
        LOG = ""