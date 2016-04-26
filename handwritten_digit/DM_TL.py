import pandas as pd
import numpy as np
from numpy.linalg import inv
import lasagne
from nolearn.lasagne import TrainSplit
import theano.tensor as T
import matplotlib.pyplot as plt
from lasagne.layers import InputLayer, DenseLayer, get_all_params
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import visualize
import os
import theano
from theano.tensor import dot, inv
from NeuralNet import NeuralNet
from CutLayer import CutLayer

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
    return -T.sum(targets*T.log(predictions) + (one-targets)*T.log(one-predictions), axis=1)

# def importance(W, last_layer, init):
#     cur_shared = []
#     (ni, no) = W.shape
#     for o in range(no):
#         abs_weight = abs(W[:,o])
#         norm = sum(abs_weight)
#         abs_weight = abs_weight / norm
#         if last_layer is None:
#             last_layer = init
#         cur_shared += [np.dot(abs_weight, last_layer)]
#     shared = sum(cur_shared)/no
#     return (shared, cur_shared)

def importance(W, last_layer, init):
    W = np.abs(W)
    norm = W.sum(axis=0)
    W = W/norm
    if last_layer is None:
        last_layer = init
    return T.dot(W, last_layer)

def select_mono(shared):
    change = [(shared[i]-shared[i-1]) for i in range(1,len(shared))]
    if sum(change) > 0:
        shared = 1-np.array(shared)
    return shared

def importance_F(W, last_layer, init):
    cur_shared = []
    (ni, no) = W.shape
    for o in range(no):
        abs_weight = np.array([abs(W[i][o]) for i in range(ni)])
        norm = sum(abs_weight)
        abs_weight /= norm
        if last_layer is None:
            last_layer = init
        cur_shared += [np.dot(abs_weight, last_layer)]
    shared = sum(cur_shared)/no
    return (shared, cur_shared)
def select_mono_F(shared):
    change = [(shared[i]-shared[i-1]) for i in range(1,len(shared))]
    if sum(change) > 0:
        shared = 1-np.array(shared)
    return shared
def calc_shared_F(net):
    shared_all_layer_I, shared_all_layer_O = [], []
    params = net.get_all_params_values()
    mid_layers = params.keys()[1:]
    shared_I, shared_O = [], []
    last_layer_I, last_layer_O = None, None
    init_I = np.array([1 for i in range(28*28)] + [0 for i in range(28*28)])
    init_O = np.array([1 for i in range(10)] + [0 for i in range(10)])
    # forward
    for layer in mid_layers:
        W = np.array(params[layer][0][:][:]) # input by output
        si, cur_shared = importance_F(W, last_layer_I, init_I)
        shared_I += [si]
        last_layer_I = np.array(cur_shared)
        shared_all_layer_I.append(cur_shared)
    mid_layers.reverse()
    # backward
    for layer in mid_layers:
        W = np.array(params[layer][0][:][:])
        trW = W.transpose()
        invW = np.dot(trW, np.linalg.pinv(np.dot(W, trW))) # output by input
        so, cur_shared = importance_F(invW, last_layer_O, init_O)
        shared_O += [so]
        last_layer_O = np.array(cur_shared)
        shared_all_layer_O.append(cur_shared)
    # select monotonically increased ones
    shared_I, shared_O = select_mono_F(shared_I), select_mono_F(shared_O)
    return shared_all_layer_I, shared_all_layer_O[::-1], shared_I, shared_O[::-1]


# def calc_shared(params):
#     shared_all_layer_I, shared_all_layer_O = [], []
#     mid_layers = params[1:]
#     shared_I, shared_O = [], []
#     last_layer_I, last_layer_O = None, None
#     init_I = np.array([1 for i in range(28*28)] + [0 for i in range(28*28)])
#     init_O = np.array([1 for i in range(10)] + [0 for i in range(10)])
#     Ws = []
#     # forward
#     for layer in mid_layers:
#         Ws += [layer[0].get_value()]
#         si, cur_shared = importance(Ws[-1], last_layer_I, init_I)
#         shared_I += [si]
#         last_layer_I = np.array(cur_shared)
#         shared_all_layer_I.append(cur_shared)
#     mid_layers.reverse()
#     Ws.reverse()
#     # backward
#     for l in range(len(mid_layers)):
#         W = Ws[l]
#         invW = dot(W.T, inv(dot(W, W.T)))
#         so, cur_shared = importance(invW.eval(), last_layer_O, init_O)
#         shared_O += [so]
#         last_layer_O = np.array(cur_shared)
#         shared_all_layer_O.append(cur_shared)
#     # select monotonically increased ones
#     shared_I, shared_O = select_mono(shared_I), select_mono(shared_O)
#     return shared_all_layer_I, shared_all_layer_O[::-1], shared_I, shared_O[::-1]

# def custom_regularizor(layers):
#     params = [x.get_params() for x in layers.values()]
#     SI, SO, si, so = calc_shared(params)
#     SI = np.array([np.var(x) for x in SI])
#     SO = np.array([np.var(x) for x in SO])
#     SI = np.exp(sum([SI[x]-SI[x-1] for x in range(1,len(SI))]))
#     SO = np.exp(sum([SO[x-1]-SO[x-1] for x in range(1,len(SO))]))
#     print(SI)
#     print(SO)
#     return (SI+SO)*LAMBDA


def calc_shared(params):
    shared_all_layer_I, shared_all_layer_O = [], []
    mid_layers = params[1:]
    last_layer_I, last_layer_O = None, None
    init_I = theano.shared(np.array([1 for i in range(28*28)] + [0 for i in range(28*28)], dtype=theano.config.floatX))
    init_O = theano.shared(np.array([1 for i in range(10)] + [0 for i in range(10)], dtype=theano.config.floatX))
    Ws = []
    # forward
    for layer in mid_layers:
        Ws += [layer]
        cur_shared = importance(Ws[-1], last_layer_I, init_I)
        last_layer_I = cur_shared
        shared_all_layer_I.append(T.var(cur_shared))
    mid_layers.reverse()
    Ws.reverse()
    # backward
    for l in range(len(mid_layers)):
        W = Ws[l]
        invW = dot(W.T, inv(dot(W, W.T)))
        cur_shared = importance(invW.eval(), last_layer_O, init_O)
        last_layer_O = cur_shared
        shared_all_layer_O.append(T.var(cur_shared))
    return shared_all_layer_I, shared_all_layer_O[::-1]
    
def custom_regularizor(layers):
    params = get_all_params(layers.values())
    params = [params[i] for i in range(0,len(params), 2)]
    print(params)
    SI, SO = calc_shared(params)
    print(SI)
    print(SO)
    SI = np.exp(sum([SI[x]-SI[x-1] for x in range(1,len(SI))]))
    SO = np.exp(sum([SO[x-1]-SO[x-1] for x in range(1,len(SO))]))
    return (SI+SO)*LAMBDA

def control_layer_num(n, l):
    tl = l
    for i in range(n):
        tl = CutLayer(tl, p=0.5)
        tl = DenseLayer(tl, num_units=HN, nonlinearity=None)
    return tl


def NN(epoch):
    l = InputLayer(name='input', shape=(None,1,28,28*2))
    l = control_layer_num(LN, l)
    l = DenseLayer(l, num_units=20, nonlinearity=lasagne.nonlinearities.sigmoid)
    net = NeuralNet(l,
                    update = nesterov_momentum,
                    update_learning_rate = 1e-5,
                    update_momentum = 0.9,
                    max_epochs = epoch,
                    verbose = 1,
                    train_split = TrainSplit(eval_size=0.2),
                    regression = True,
                    objective_loss_function = multilabel_objective,
                    custom_regularizor = custom_regularizor
    )
    return net


def proc(Adf, Bdf, A, B):
    ORIGIN_HEADER = ['pixel{0}'.format(x) for x in range(28*28)]
    ORIGIN_HEADER += ['label']
    HEADER = ['{0}_{1}'.format(domain, column) for domain in ['A','B'] for column in ORIGIN_HEADER]
    df_train = pd.DataFrame({x:[]} for x in HEADER)
    for aidx, a in Adf.iterrows():
        for bidx, b in Bdf.iterrows():
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
    return df

def gen_data(A, B, split_ratio, num, filename):
    A_train, A_test = get_classes(A, SPLIT_RATIO, NUM)
    B_train, B_test = get_classes(B, SPLIT_RATIO, NUM)
    df = proc(A_train, B_train, A, B)
    df.to_csv('{0}_train.csv'.format(filename))
    df = proc(A_test, B_test, A, B)
    df.to_csv('{0}_test.csv'.format(filename))

def gen_data_label(filename):
    df_train = pd.read_csv(filename)
    label = []
    for idx, row in df_train.iterrows():
        l = np.zeros(20)
        l[int(row['A_label'])] = 1
        l[10+int(row['B_label'])] = 1
        label += [l]

    df_train['label'] = label
    df_train.to_csv('combined_{0}'.format(filename))
    HEADER = ['label']
    HEADER += ['A_pixel{0}'.format(x) for x in range(28*28)]
    HEADER += ['B_pixel{0}'.format(x) for x in range(28*28)]
    train = df_train[HEADER[1:]].values
    train = np.array(train).reshape((-1,1,28,28*2)).astype(np.uint8)
    label = np.array(label).astype(np.uint8)
    return train, label

def select_max(x):
    A, B = np.argmax(x[:10]), np.argmax(x[10:])
    return np.array([1 if i==A or i==10+B else 0 for i in range(20)])

def plot_all_layer_shared_dist(shared_all_layer, filename):
    if not os.path.exists(filename):
        os.makedirs(filename)
    l = len(shared_all_layer)
    for i in range(l):
        plt.hist(shared_all_layer[i])
        plt.title('layer_{0}'.format(i+1))
        plt.savefig(os.path.join(filename,'layer_{0}'.format(i+1)))
        plt.hold(False)

def run(filename):
    train, train_label = gen_data_label('{0}_train.csv'.format(filename))
    test, test_label = gen_data_label('{0}_test.csv'.format(filename))
    while True:
        net = NN(EPOCH)
        net.fit(train, train_label)
        shared_all_layer_I, shared_all_layer_O, shared_I, shared_O = calc_shared_F(net)
        pred = net.predict(test)
        n = len(pred)
        acc = 0
        for i in range(n):
            p = select_max(pred[i])
            compare = [0 if p[x]==test_label[i][x] else 1 for x in range(20)]
            acc = acc+1 if sum(compare) == 0 else acc
        accuracy = float(acc)/n
        print('accuracy={0}'.format(accuracy))
        if accuracy > ACC:
            break
    # plot shared_input
    plt.boxplot(shared_all_layer_I)
    plt.title('[{0} forward] acc={3}%\n{1} test instances, {2} correct'.format(filename, n,acc,100*accuracy))
    plt.savefig('F_n={3},l={1},pn={2},{0}'.format(filename,LN,HN,NUM))
    plt.hold(False)
    # plot shared_output
    plt.boxplot(shared_all_layer_O)
    plt.title('[{0} backward] acc={3}%\n{1} test instances, {2} correct'.format(filename, n,acc,100*accuracy))
    plt.savefig('B_n={3},l={1},pn={2},{0}'.format(filename,LN,HN,NUM))
    plt.hold(False)
    return net

SPLIT_RATIO = 0.9
NUM = 10
LN = 10
HN = 50
LAMBDA = 1
ACC = 0.85
EPOCH = 300

fname = 'low'
A, B = [1,7,4], [0,9,6]
# gen_data(A,B,SPLIT_RATIO,NUM,fname)
net = run(fname)

fname = 'mid'
A, B = [1,0,4], [7,9,6]
# gen_data(A,B,SPLIT_RATIO,NUM,fname)
run(fname)

fname = 'high'
A, B = [1,7,0], [0,1,9]
# gen_data(A,B,SPLIT_RATIO,NUM,fname)
run(fname)