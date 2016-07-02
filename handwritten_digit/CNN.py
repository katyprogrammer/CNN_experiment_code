import sys
import os
from os.path import exists, join
import time
import cPickle
import optparse

import numpy as np
import scipy.sparse as Sp
from sklearn.datasets import fetch_mldata
from sktensor import dtensor, cp_als
import theano
import theano.tensor as T
import lasagne

'''
### tuning good hyperparameter using all category ###
training all:
$ python CNN.py -e 20 > all.txt

### training domain A(src) ###
# training with 20 epoch, save trained params to A.pkl
$ python CNN.py -r A -d A.pkl -e 20 > A.txt
# training domain B(tgt), load trained params from A.pkl
$ python CNN.py -r B -l A.pkl -d B.pkl -e 20 > B.txt
# load low-rank with R rank1 from A.pkl
$ python CNN.py -r B -l A.pkl -d B_1.pkl -e 20 -R 1 > B_1.txt

### A.txt, B.txt will contain training information(training error, validation error, validation accuracy)
$ python plot.py -i A.txt -o A
$ python plot.py -i B.txt -o B

### using GPU
# on all 1 epoch:
# cpu => 220s, gpu => 26s
$ THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python CNN.py ...
'''

# src
A = np.array(range(5))
# tgt
B = 5+A

isDownload = True
BATCHN = 500
LAYERS = ['conv_1','maxpool_1','conv_2','maxpool_2','dropout_1','dense_1','dropout_2','dense_output']

def parse_arg():
    parser = optparse.OptionParser('usage%prog [-l load parameterf from] [-d dump parameter to] [-e epoch] [-r src or tgt]')
    parser.add_option('-l', dest='fin')
    parser.add_option('-d', dest='fout')
    parser.add_option('-e', dest='epoch')
    parser.add_option('-r', dest='A_B')
    parser.add_option('-R', dest='rank')
    (options, args) = parser.parse_args()
    return options

def load_dataset(A_B):
    def download_save_by_category():
        mnist = fetch_mldata('MNIST original')
        # mnist.data (70000, 784), mnist.target (70000, 1)
        trainX, trainY = mnist.data[:-10000], mnist.target[:-10000]
        testX, testY = mnist.data[-10000:], mnist.target[-10000:]
        if not exists('train'):
            os.makedirs('train')
        x = {i:[] for i in range(10)}
        for i in range(len(trainY)):
            tmp = x[trainY[i]]
            tmp.append(trainX[i])
            x[trainY[i]] = tmp
        for i in range(10):
            cPickle.dump(x[i], open(join('train', '{}.pkl'.format(i)), 'w+'))
        if not exists('test'):
            os.makedirs('test')
        x = {i:[] for i in range(10)}
        for i in range(len(testY)):
            tmp = x[testY[i]]
            tmp.append(testX[i])
            x[testY[i]] = tmp
        for i in range(10):
            cPickle.dump(x[i], open(join('test', '{}.pkl'.format(i)), 'w+'))
    def read_and_split(filepath, digit, NUM=None, Split=True):
        data = cPickle.load(open(filepath, 'r'))
        # number of instance to use
        if NUM is not None:
            data = data[:NUM]
        target = [digit for i in range(len(data))]
        if not Split:
            return data, target
        # split to train/valid
        valid_ration = 0.2
        split = int(len(target)*valid_ration)
        valid, valid_tgt = data[:split], target[:split]
        train, train_tgt = data[split:], target[split:]
        return train, valid, train_tgt, valid_tgt
    def get_classes(classes):
        train, valid, test, train_tgt, valid_tgt, test_tgt = None, None, None, None, None, None
        for digit in classes:
            tr, v, trt, vt = read_and_split('train/{}.pkl'.format(digit), digit)
            te, tet = read_and_split('test/{}.pkl'.format(digit), digit, Split=False)
            train, train_tgt = (tr, trt) if train is None else (np.vstack([train,tr]), np.hstack([train_tgt, trt]))
            valid, valid_tgt = (v, vt) if valid is None else (np.vstack([valid, v]), np.hstack([valid_tgt, vt]))
            test, test_tgt = (te, tet) if test is None else (np.vstack([test, te]), np.hstack([test_tgt, tet]))
        return train, valid, test, train_tgt, valid_tgt, test_tgt

    if not isDownload:
        download_save_by_category()
    classes = A if A_B == 'A' else B
    if A_B is None:
        classes = range(10)
    return get_classes(classes)
####### Build the neural network model #######
# A function that takes a Theano variable representing the input and returns
# the output layer of a neural network model built in Lasagne.
def build_mlp(input_var=None):
    l_in = lasagne.layers.InputLayer(name='input', shape=(None, 1, 28, 28), input_var=input_var)
    conv1 = lasagne.layers.Conv2DLayer(
        l_in, name='conv_1', num_filters=32, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform()
        )
    maxpool1 = lasagne.layers.MaxPool2DLayer(
        conv1, name='maxpool_1', pool_size=(2, 2)
    )
    conv2 = lasagne.layers.Conv2DLayer(
        maxpool1, name='conv_2', num_filters=32,
        filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify
    )
    maxpool2 = lasagne.layers.MaxPool2DLayer(
        conv2, name='maxpool_2', pool_size=(2, 2)
    )
    dropout1 = lasagne.layers.DropoutLayer(
        maxpool2, name='dropout_1', p=0.5
    )
    dense = lasagne.layers.DenseLayer(
        dropout1, name='dense_1', num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify
    )
    dropout2 = lasagne.layers.DropoutLayer(
        dense, name='dropout_2', p=0.5
    )
    l_out = lasagne.layers.DenseLayer(
        dropout2, name='dense_output', num_units=10,
        nonlinearity=lasagne.nonlinearities.softmax,
    )
    return l_out

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

# CP low-rank approximate
def approx_CP_R(value, R):
    if value.ndim < 2:
        return value
    T = dtensor(value)
    P, fit, itr, exetimes = cp_als(T, R, init='random')
    Y = None
    for i in range(R):
        y = P.lmbda[i]
        o = None
        for l in range(T.ndim):
            o = P.U[l][:,i] if o is None else np.outer(o, P.U[l][:,i])
        y = y * o
        Y = y if Y is None else Y+y
    return Y
def load_largest_rank(A, O, rank):
    def all_layers(A, O, TARGET):
        AA, BB = [], []
        row, column = -1, -1
        for i in range(len(A)):
            if A[i].name not in TARGET:
                continue
            if 'dense' in A[i].name or 'conv' in A[i].name:
                w, b = A[i].get_params()
                w, b = w.get_value(), b.get_value()
                print('w={},b={}'.format(w.shape, b.shape))
                BB.append(b)
                c = 1
                for j in range(1,w.ndim):
                    c*=w.shape[j]
                T = w.reshape((w.shape[0],c))
                AA.append(T)
                row, column = max(row, len(T)), max(column, len(T[1]))
            else:
                print('layers other than "dense", "conv" layers, donnot have params')
            
        for i in range(len(AA)):
            ac = len(AA[i][0])
            while row > len(AA[i]):
                AA[i] = np.append(AA[i], np.zeros((1,ac)), axis=0)
            if column > ac:
                TAA = []
                for j in range(row):
                    TAA += [np.append(AA[i][j], np.zeros(column-ac))]
                AA[i] = np.array(TAA)
        for i in range(len(BB)):
            ac = len(BB[i])
            if column > ac:
                BB[i] = np.append(BB[i], np.zeros(column-ac))
        return np.array(AA), np.array(BB)
    def de_all_layers(O, AA, BB, TARGET):
        A = []
        ai = 0
        for i in range(len(O)):
            if O[i].name not in TARGET:
                x = O[i].get_params()
                if x != []:
                    w, b = x
                    A.append(w.get_value())
                    A.append(b.get_value())
            else:
                w, b = O[i].get_params()
                w, b = w.get_value(), b.get_value()
                c = 1
                for j in range(1,w.ndim):
                    c*=w.shape[j]
                TAA = []
                X = AA[ai]
                X = X[:w.shape[0]]
                for j in range(X.shape[0]):
                    TAA += [X[j][:c]]
                A.append(np.array(TAA).reshape((w.shape)))
                A.append(BB[ai][:len(b)])
                ai += 1
        return A
    ### tensorization ###
    TARGET = ['conv_1','conv_2']
    TAA, TBB = all_layers(A, O, TARGET)
    print('decomposing tensor W of shape {}...'.format(TAA.shape))
    print('decomposing tensor B of shape {}...'.format(TBB.shape))
    AA = approx_CP_R(TAA, int(rank)).reshape(TAA.shape)
    BB = approx_CP_R(TBB, int(rank)).reshape(TBB.shape)
    ### de-tensorization ###
    # all layers
    A = de_all_layers(O, AA, BB, TARGET)
    print(len(A))
    return A
def load_smallest_rank(A, O, rank):
    TA = load_largest_rank(A, O, rank)
    return np.array(A)-np.array(TA)+np.array(O)

def main(num_epochs=500,fin_params=None,fout_params=None,A_B=None, rank=None):
    def debug_theano_var(var, outf):
        if not exists('debug'):
            os.makedirs('debug')
        theano.printing.pydotprint(var, outfile=join('debug', outf))
    
    ### load dataset ###
    print('loading data...')
    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(A_B)
    X_train, X_val, X_test = X_train.reshape(-1,1,28,28)/np.float32(256), X_val.reshape(-1,1,28,28)/np.float32(256), X_test.reshape(-1,1,28,28)/np.float32(256)
    print('#train = {0}, #test = {1}, #valid = {2}'.format(len(y_train), len(y_test), len(y_val)))

    # prepare theano variables
    input_var = T.tensor4('inputs')
    target_var = T.lvector('targets')

    ### create neural network model ###
    print("building model and compiling functions...")
    network = build_mlp(input_var)

    # load parameters
    if fin_params is not None:
        A = cPickle.load(open(fin_params, 'r'))
        O = lasagne.layers.get_all_layers(network)
        if rank is not None:
            # try largest 1-rank approximate
            A = load_largest_rank(A, O, rank)
            # try largest 1-rank approximate
            # A = load_smallest_rank(A, O, rank)
        lasagne.layers.set_all_param_values(network, A)
    params = lasagne.layers.get_all_params(network, trainable=True)

    ### Create objective expression for training ###
    prediction = lasagne.layers.get_output(network)
    # debug_theano_var(prediction, 'forward.png')
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # debug_theano_var(loss, 'loss.png')
    # We could add some weight decay as well here, see lasagne.regularization.

    ### set training update expression: SGD with Nesterov momentum ###
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=1e-3, momentum=0.9
    )
    ## debug print
    # c = 0
    # for k,v in updates.items():
    #     debug_theano_var(v, 'sgd_momentum_{1}{0}.png'.format(k, c))
    #     c+=1

    # loss expression for validation/testing
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)
    # compile theano function
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    ### Mini-Batch Training Procedure ###
    print('starting training...')
    for epoch in range(num_epochs):
        train_err, train_batches = 0, 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, BATCHN, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1
        val_err, val_acc, val_batches = 0, 0, 0
        for batch in iterate_minibatches(X_val, y_val, BATCHN, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1
        print('Epoch {} of {} took {:.3f}s'.format(epoch+1, num_epochs, time.time()-start_time))
        print('  training loss:\t\t{:.6f}'.format(train_err/train_batches))
        print('  validation loss:\t\t{:.6f}'.format(val_err/val_batches))
        print('  validation accuracy:\t\t{:.2f} %'.format(val_acc/val_batches*100))
    # after training, compute test error
    test_err, test_acc, test_batches = 0, 0, 0
    for batch in iterate_minibatches(X_test, y_test, BATCHN, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print('Final results:')
    print('  test loss:\t\t\t{:.6f}'.format(test_err/test_batches))
    print('  test accuracy:\t\t{:.2f} %'.format(test_acc/test_batches*100))

    # dump parameters
    cPickle.dump(lasagne.layers.get_all_layers(network), open(fout_params, 'w+'))

if __name__ == '__main__':
    opts = parse_arg()
    kwargs = {}
    if len(sys.argv) > 1:
        kwargs['num_epochs'] = int(opts.epoch)
        kwargs['fin_params'] = opts.fin
        kwargs['fout_params'] = opts.fout
        kwargs['A_B'] = opts.A_B
        kwargs['rank'] = opts.rank

    main(**kwargs)