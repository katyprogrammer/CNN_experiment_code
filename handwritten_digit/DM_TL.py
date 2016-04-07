import pandas as pd
import numpy as np
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize

# def NN(epoch):
#     net = NeuralNet(layers=[
#     ('input', layers.InputLayer), ('hidden', layers.DenseLayer), ('hidden', layers.DenseLayer)]),