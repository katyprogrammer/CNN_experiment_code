import pandas as pd
import numpy as np
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize

def NN(epoch):
    net1 = NeuralNet(layers=[
    ('input', layers.InputLayer), ('conv', layers.Conv2DLayer), ('pool', layers.MaxPool2DLayer), ('hidden', layers.DenseLayer), ('output', layers.DenseLayer)],
                     input_shape = (None,1,28,28),
                     conv_num_filters = 7, conv_filter_size = (3,3), conv_nonlinearity = lasagne.nonlinearities.rectify,
                     pool_pool_size = (2,2),
                     hidden_num_units = 1000,
                     output_nonlinearity = lasagne.nonlinearities.softmax,
                     output_num_units = 10,
                     update = nesterov_momentum,
                     update_learning_rate = 0.0001,
                     update_momentum = 0.9,
                     max_epochs = epoch,
                     verbose = 1
    )
    return net1

dataset = pd.read_csv('input/train.csv')
label = dataset[[0]].values.ravel()
train = dataset.iloc[:,1:].values
test = pd.read_csv('input/test.csv').values

label = label.astype(np.uint8)
train = np.array(train).reshape((-1, 1, 28, 28)).astype(np.uint8)
test = np.array(test).reshape((-1, 1, 28, 28)).astype(np.uint8)
net1 = NN(50)
net1.fit(train, label)
pred = net1.predict(test)
np.savetxt('submission_nn.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header='ImageId,Label', comments='', fmt='%d')