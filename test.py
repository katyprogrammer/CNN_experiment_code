import sys
import os
from os.path import exists, join
import time
import cPickle
import optparse

import numpy as np
import scipy.sparse as Sp
from sklearn.datasets import fetch_mldata
# from sktensor import dtensor, cp_als
import theano
import theano.tensor as T
import lasagne
import random
# python test.py -r A -d A.pkl -e 20 -s small -t 1 2 3 4 5 > A.txt
def parse_arg():
    parser = optparse.OptionParser('usage%prog [-l load parameterf from] [-d dump parameter to] [-e epoch] [-r src or tgt] [-small small or large rank] [-t 1 2 3 4]')
    parser.add_option('-l', dest='fin')
    parser.add_option('-d', dest='fout')
    parser.add_option('-e', dest='epoch')
    parser.add_option('-r', dest='A_B')
    parser.add_option('-R', dest='rank')
    parser.add_option('-s', dest='small')
    parser.add_option('-t', dest='transfer')
    (options, args) = parser.parse_args()
    return options

def main(num_epochs=100,fin_params=None,fout_params=None,A_B=None, rank=None, small=None, transfer=None):
    print "hi"
    print transfer
    print fin_params
    print num_epochs


if __name__ == '__main__':
    opts = parse_arg()
    kwargs = {}
    if len(sys.argv) > 1:
        kwargs['num_epochs'] = int(opts.epoch)
        kwargs['fin_params'] = opts.fin
        kwargs['fout_params'] = opts.fout
        kwargs['A_B'] = opts.A_B
        kwargs['rank'] = opts.rank
        kwargs['small'] = opts.small
        kwargs['transfer'] = opts.transfer
        print "parsing argument..."
        print opts.transfer
        print sys.argv
        print len(sys.argv)
        print sys.argv.index("-t")
        transfer_array = sys.argv[sys.argv.index("-t")+1:]
        print transfer_array
    main(**kwargs)
