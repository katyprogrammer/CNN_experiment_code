import matplotlib.pyplot as plt
import cPickle
import glob
import os
import numpy as np

files = glob.glob('*.pkl')
if not os.path.exists('conv1'):
    os.makedirs('conv1')
if not os.path.exists('conv2'):
    os.makedirs('conv2')
for f in files:
    name = f.split('.')[0]
    P = cPickle.load(open('{}.pkl'.format(name), 'r'))
    P1 = P[1].get_params()[0].get_value() # (32,1,5,5)
    plt.imshow(P1.reshape(4,8,5,5).swapaxes(1,2).reshape(4*5,-1), cmap=plt.get_cmap('gray'))
    plt.savefig(os.path.join('conv1','{}.png'.format(name)))
    P2 = P[3].get_params()[0].get_value() # (32,32,5,5)
    plt.imshow(P2.reshape(32,32,5,5).swapaxes(1,2).reshape(32*5,-1), cmap=plt.get_cmap('gray'))
    plt.savefig(os.path.join('conv2','{}.png'.format(name)))