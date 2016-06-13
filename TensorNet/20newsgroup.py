#!/usr/bin/env python

from __future__ import print_function

import sys
import os
from os.path import exists, join
import time
import cPickle
import optparse

import numpy as np
import scipy.sparse as Sp
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sktensor import dtensor, cp_als
import theano
import theano.tensor as T
import lasagne

from ttlayer import TTLayer

'''
# tuning good hyperparameter using all category
training all:
$ python 20newsgroup.py -e 5000 > all.txt

training domain A(src)
# training with 5000 epoch
# save trained params to A.pkl
$ python 20newsgroup.py -r A -d A.pkl -e 5000 > A.txt
training domain B(tgt)
# load trained params from A.pkl
$ python 20newsgroup.py -r B -l A.pkl -d B.pkl -e 5000 > B.txt
# load low-rank with R rank1 from A.pkl
$ python 20newsgroup.py -r B -l A.pkl -d B_1.pkl -e 5000 -R 1 > B_1.txt

A.txt, B.txt will contain training information(training error, validation error, validation accuracy)
$ python plot.py -i A.txt -o A
$ python plot.py -i B.txt -o B
'''

def parse_arg():
    parser = optparse.OptionParser('usage%prog [-l load parameterf from] [-d dump parameter to] [-e epoch] [-r src or tgt]')
    parser.add_option('-l', dest='fin')
    parser.add_option('-d', dest='fout')
    parser.add_option('-e', dest='epoch')
    parser.add_option('-r', dest='A_B')
    parser.add_option('-R', dest='rank')
    (options, args) = parser.parse_args()
    return options

# src
A = np.array(range(10))
# tgt
B = 10+A

BATCH_EPOCH = 20
ROWN, BATCHN = None, None
HashN = 200
isDownload = True

def load_dataset(A_B):
    def download_save_by_category():
        # download 20newsgroups
        newsgroups = fetch_20newsgroups(subset='all')
        global ROWN
        ROWN = len(newsgroups.target)
        global BATCHN
        BATCHN = ROWN / BATCH_EPOCH
        global HashN
        first = True
        TfIdfVec = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
        vectors = TfIdfVec.fit_transform(newsgroups.data) # in scipy.sparse.csr_matrix format
        lda = TruncatedSVD(n_components=HashN) # sparse linear discriminant analysis
        vectors = lda.fit_transform(vectors, newsgroups.target)
        
        target = newsgroups.target
        print('[LDA] transformed dimension={0}'.format(vectors.shape[1]))
        # feature hashing
        # hasher = HashingVectorizer(n_features=HashN)
        # vectors = hasher.fit_transform(newsgroups.data)
        # target = newsgroups.target
        HashN = vectors.shape[1]

        if not exists('train'):
            os.makedirs('train')

        for i in range(20):
            x = []
            for j in range(len(target)):
                if target[j] == i:
                    x.append(vectors[j])
            cPickle.dump(x, open(join('train', '{0}.pkl'.format(i)), 'w+'))
    
    def read_and_split(filepath, digit, NUM=None, test_ratio=0.2, valid_ratio=0.2):
        data = cPickle.load(open(filepath, 'r'))
        # instance number to use
        if NUM is not None:
            data = data[:NUM]
        # split train/test
        split = int(len(data)*(1-test_ratio))
        train = data[:split]
        test = data[split:]
        train_tgt = [digit for i in range(len(train))]
        test_tgt = [digit for i in range(len(test))]
        # split train/valid
        split = int(len(train_tgt)*valid_ratio)
        valid = train[:split]
        train = train[split:]
        valid_tgt = train_tgt[:split]
        train_tgt = train_tgt[split:]
        return Sp.csr_matrix(train), Sp.csr_matrix(valid), Sp.csr_matrix(test), train_tgt, valid_tgt, test_tgt

    def get_classes(classes):        
        train, valid, test, train_tgt, valid_tgt, test_tgt = None, None, None, None, None, None
        for digit in classes:
            tr, v, te, trt, vt, tet = read_and_split('train/{0}.pkl'.format(digit), digit)
            if train is None:
                train, train_tgt = (Sp.vstack(tr), trt) if tr is not None and len(trt)>0 else (train, train_tgt)
            else:
                train, train_tgt = (Sp.vstack([train, Sp.vstack(tr)]), train_tgt+trt) if tr is not None and len(trt)>0 else (train, train_tgt)
            if valid is None:
                valid, valid_tgt = (Sp.vstack(v), vt) if v is not None and len(vt)>0 else (valid, valid_tgt)
            else:
                valid, valid_tgt = (Sp.vstack([valid, Sp.vstack(v)]), valid_tgt+vt) if v is not None and len(vt)>0 else (valid, valid_tgt)
            if test is None:
                test, test_tgt = (Sp.vstack(te), tet) if te is not None and len(tet)>0 else (test, test_tgt)
            else:
                test, test_tgt = (Sp.vstack([test, Sp.vstack(te)]), test_tgt+tet) if te is not None and len(tet)>0 else (test, test_tgt)
        global ROWN
        ROWN = len(train_tgt)+len(valid_tgt)+len(test_tgt)
        global BATCHN
        BATCHN = ROWN / BATCH_EPOCH
        
        return train, valid, test, np.array(train_tgt), np.array(valid_tgt), np.array(test_tgt)

    if not isDownload:
        download_save_by_category()
    if A_B == 'A':
        classes = A
    elif A_B == 'B':
        classes = B
    else:
        classes = range(20)
    return get_classes(classes)


# ##################### Build the neural network model #######################
# A function that takes a Theano variable representing the input and returns
# the output layer of a neural network model built in Lasagne.

def build_mlp(input_var=None):
    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 1 row, HashN columns) and
    # linking it to the given Theano variable `input_var`, if any:
    l_in = lasagne.layers.InputLayer(shape=(None, HashN),
                                     input_var=input_var)

    # Build a TT-layer
    # l_hid1 = TTLayer(
    #         l_in, tt_input_shape=[HashN], tt_output_shape=[HashN],
    #         tt_ranks=[1, 1],
    #         nonlinearity=lasagne.nonlinearities.tanh)

    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_in, num_units=1000,
            nonlinearity=None)

    # Finally, we'll add the fully-connected output layer, of 20 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid2, num_units=20,
            nonlinearity=lasagne.nonlinearities.softmax)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out

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

# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize] if shuffle else slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(num_epochs=500,fin_params=None,fout_params=None,A_B=None, rank=None):
    # Load the dataset
    print("Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(A_B)
    X_train, X_val, X_test = X_train.todense(), X_val.todense(), X_test.todense()
    print('#train = {0}, #test = {1}, #valid = {2}'.format(len(y_train), len(y_test), len(y_val)))
    # Prepare Theano variables for inputs and targets
    input_var = T.matrix('inputs')
    target_var = T.lvector('targets')

    # Create neural network model.
    print("Building model and compiling functions...")
    network = build_mlp(input_var)

    # Create a loss expression for training
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Load parameters
    if fin_params is not None:
        A = cPickle.load(open(fin_params, 'r'))
        if rank is not None:
            Ap = []
            for i in range(len(A)):
                Ap.append(approx_CP_R(A[i], int(rank)))
            A = Ap
        lasagne.layers.set_all_param_values(network, A)
    params = lasagne.layers.get_all_params(network, trainable=True)
     # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=1e-3, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, BATCHN, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, BATCHN, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, BATCHN, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    # dump parameters
    if fout_params is None:
        fout_params = 'model_{0}.npz'.format(num_epochs)
    cPickle.dump(lasagne.layers.get_all_param_values(network), open(fout_params, 'w+'))


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