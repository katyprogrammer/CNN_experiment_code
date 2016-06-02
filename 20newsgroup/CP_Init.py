import pandas as pd
import numpy as np
import lasagne
from nolearn.lasagne import TrainSplit
import theano.tensor as T
import matplotlib.pyplot as plt
from lasagne.layers import DenseLayer, InputLayer
from lasagne.updates import nesterov_momentum, adagrad
from NeuralNet import NeuralNet
import theano.sparse as S
import scipy.sparse as Sp
import os
from os.path import join
import time
import cPickle

LOG = ""
LEN = None

# data processing
def read_and_split(filepath, digit):
    data = cPickle.load(open(filepath, 'r'))
    if NUM is not None:
        data = data[:NUM]
    split = int(len(data)*SPLIT_RATIO)
    train = data[:split]
    test = data[split:]
    train_target = [digit for i in range(len(train))]
    test_target = [digit for i in range(len(test))]
    return train, test, train_target, test_target

def get_classes(classes):
    train, test, train_tgt, test_tgt = None, None, None, None
    for digit in classes:
        tr, te, trt, tet = read_and_split('train/{0}.pkl'.format(digit), digit)
        if train is None:
            train, test, train_tgt, test_tgt = Sp.vstack(tr), Sp.vstack(te), trt, tet
        else:
            train = Sp.vstack([train, Sp.vstack(tr)])
            test = Sp.vstack([test, Sp.vstack(te)])
            train_tgt += trt
            test_tgt += tet
    return train, test, np.array(train_tgt), np.array(test_tgt)

def gen_data(A, B):
    A_train, A_test, A_train_tgt, A_test_tgt = get_classes(A)
    B_train, B_test, B_train_tgt, B_test_tgt = get_classes(B)

    global LOG
    LOG += "\ttrain\ttest\nA\t{0}\t{1}\nB\t{2}\t{3}\n".format(len(A_train_tgt), len(A_test_tgt), len(B_train_tgt), len(B_test_tgt))
    cPickle.dump(A_train, open((join(RUN_NAME, '{0}_A_train.pkl'.format(RUN_NAME))), 'w+'))
    cPickle.dump(A_test, open((join(RUN_NAME, '{0}_A_test.pkl'.format(RUN_NAME))), 'w+'))
    cPickle.dump(A_train_tgt, open((join(RUN_NAME, '{0}_A_train_tgt.pkl'.format(RUN_NAME))), 'w+'))
    cPickle.dump(A_test_tgt, open((join(RUN_NAME, '{0}_A_test_tgt.pkl'.format(RUN_NAME))), 'w+'))
    cPickle.dump(B_train, open((join(RUN_NAME, '{0}_B_train.pkl'.format(RUN_NAME))), 'w+'))
    cPickle.dump(B_test, open((join(RUN_NAME, '{0}_B_test.pkl'.format(RUN_NAME))), 'w+'))
    cPickle.dump(B_train_tgt, open((join(RUN_NAME, '{0}_B_train_tgt.pkl'.format(RUN_NAME))), 'w+'))
    cPickle.dump(B_test_tgt, open((join(RUN_NAME, '{0}_B_test_tgt.pkl'.format(RUN_NAME))), 'w+'))

def read_data(filename):
    data = cPickle.load(open('{0}.pkl'.format(filename), 'r'))
    tgt = cPickle.load(open('{0}_tgt.pkl'.format(filename), 'r'))
    global LEN
    LEN = data.shape[1]
    return data.todense().reshape((-1,LEN)), tgt.astype(np.uint8)


# training
def control_layer_num(n, l):
    tl = l
    for i in range(n):
        tl = DenseLayer(tl, num_units=HN, nonlinearity=lasagne.nonlinearities.rectify)
    return tl

def NN(epoch, custom_regularizor=None):
    l = InputLayer(name='input', shape=(None,LEN))
    l = DenseLayer(l, name='input_1', num_units=HN, nonlinearity=lasagne.nonlinearities.rectify)
    l = control_layer_num(LN, l)
    l = DenseLayer(l, name='output', num_units=20, nonlinearity=lasagne.nonlinearities.softmax)
    net = NeuralNet(l,
                    update = nesterov_momentum,
                    update_learning_rate = 1e-6,
                    update_momentum = 0.9,
                    max_epochs = epoch,
                    verbose = 1,
                    train_split = TrainSplit(eval_size=0.2),
                    custom_regularizor = custom_regularizor,
    )
    return net

def save_params(A_OR_B, net):
    net.save_params_to(join(RUN_NAME, '{0}_{1}_net.pkl'.format(RUN_NAME, A_OR_B)))

def run(A_OR_B, CP_R=None, LNum=None):
    global LOG
    train, train_label = read_data(join(RUN_NAME, '{0}_{1}_{2}'.format(RUN_NAME, A_OR_B, 'train')))
    test, test_label = read_data(join(RUN_NAME, '{0}_{1}_{2}'.format(RUN_NAME, A_OR_B, 'test')))
    while True:
        net = NN(EPOCH)
        # load trained parameters
        if CP_R is not None:
            # measure CP approximate time
            st = time.time()
            if LNum is None:
                global LN
                LNum = LN
            if LNum == -1: # copy all A
                net.load_params_from(join(RUN_NAME, '{0}_{1}_net.pkl'.format(RUN_NAME, 'A')))
            else:
                net.load_CP_approx_params_from(join(RUN_NAME, '{0}_{1}_net.pkl'.format(RUN_NAME, 'A')), HN, LNum, CP_R=CP_R)
                ed = time.time()
                LOG += "CP_approximate_exetime: {0}s\n".format(ed-st)
        # measure fitting time
        st = time.time()
        # indata dimension issue
        net.fit(train, train_label)
        ed = time.time()
        LOG += "training_time: {0}s\n".format(ed-st)
        pred = net.predict(test)
        n = len(pred)
        acc = 0
        for i in range(n):
            acc = acc+1 if pred[i]==test_label[i] else acc
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
LN = 3 # layer number
HN = 1000 # hidden unit per layer
ACC = 0.8
EPOCH = 100

RUN_NAME = 'MLP_{0}LN_{1}HN'.format(LN,HN)
if not os.path.exists(RUN_NAME):
    os.makedirs(RUN_NAME)


# logging
f = open(join(RUN_NAME, 'log_{0}.txt'.format(RUN_NAME)), 'a+')
A, B = [1,7,4,5,8,11,17,14,18,15], [2,3,6,0,9,12,13,16,10,19] # AB


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
LOG += ""
EXP_NAME = 'B_Raw'
net = run('B', CP_R=-1, LNum=-1)
LOG += '---' * 9 + '\n'
f.write(LOG)

# LOG = ""
# for i in range(1,3):
#     for j in range(LN,LN+1):
#         EXP_NAME = '{0}_rank1_{1}_Layer'.format(i, j)
#         # transfer R low-rank approximation
#         net2 = run('B', CP_R=i, LNum=j)
#         LOG += '---' * 9 + '\n'
#         f.write(LOG)
#         LOG = ""
