import numpy as np
import matplotlib.pyplot as plt

def calc_shared(mid_layers):
    shared = []
    last_layer = None
    for layer in mid_layers:
        cur_shared = []
        neuron_num = len(layer)
        x_num = len(layer[0])
        for neuron in range(neuron_num):
            abs_weight = np.array([abs(layer[neuron][i]) for i in range(x_num)])
            norm = sum(abs_weight)
            abs_weight /= norm
            if last_layer is None:
                last_layer = [1 for i in range(1)]
                last_layer += [0 for i in range(5)]
                last_layer = np.array(last_layer)
            cur_shared += [np.dot(abs_weight, last_layer)]
        print(cur_shared)
        shared += [sum(cur_shared)/neuron_num]
        last_layer = np.array(cur_shared)
    # calc change
    change = [(shared[i]-shared[i-1]) for i in range(1,len(shared))]
    if sum(change) > 0:
        shared = 1-np.array(shared)
    return shared

A = [1.0,1.0,1.0,0,0,0]
B = [0,0,0,1.0,1.0,1.0]
layer1 = [A,B,B,B,B,B]
layer2 = [A,A,A,B,B,B]
layer3 = [A,A,A,A,A,B]
layers = [layer1,layer2,layer3]
shared = calc_shared(layers)
print(shared)