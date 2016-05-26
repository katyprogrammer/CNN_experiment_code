import optparse
import pandas as pd
import matplotlib.pyplot as plt
import re

def parse_arg():
    parser = optparse.OptionParser('usage%prog [-i inFile] [-o outFile]')
    parser.add_option('-i', dest='inFile', default='')
    parser.add_option('-o', dest='outFile', default='')
    (options, args) = parser.parse_args()
    return options

def plot_acc(fin, fout):
    plt.clf()
    with open(fin) as f:
        data = f.read()
    # train, test information
    A_train = re.findall('A\t(.*)\t', data)[0]
    A_test = re.findall('A\t.*\t(.*)', data)[0]
    B_train = re.findall('B\t(.*)\t', data)[0]
    B_test = re.findall('B\t.*\t(.*)', data)[0]
    # table = plt.table(cellText=[[A_train, A_test],[B_train, B_test]], rowLabels=['A','B'], colLabels=['train','test'])

    # baseline
    A_base_acc = re.findall('training_time: .*\n\[A_Baseline\] acc = (.*)', data)[-1]
    B_base_acc = re.findall('training_time: .*\n\[B_Baseline\] acc = (.*)', data)[-1]
    A_acc, = plt.plot([0], [A_base_acc], marker='o', markersize=12, label='A baseline')
    B_acc, = plt.plot([0], [B_base_acc], marker='o', markersize=12, label='B baseline')
    # Low rank
    R = set([int(x) for x in re.findall('\[.*_(\d*)_rank1_.*\]', data)])
    L = set([int(x) for x in re.findall('\[.*_(\d*)_Layer\]', data)])
    PLOT = []
    for r in R:
        x, y = [0], [B_base_acc]
        for l in L:
            acc = re.findall('\[B_{0}_rank1_{1}_Layer\] acc = (.*)'.format(r,l), data)[-1]
            x += [l]
            y += [acc]
        PLOT += [plt.plot(x, y, marker='.', markersize=5, label='{0} Rank1'.format(r))]
    PLOT += [A_acc,B_acc]
    plt.title('Accuracy')
    plt.legend()
    plt.xlabel('#transfered layers')
    plt.ylabel('accuracy')
    plt.savefig('{0}_acc.png'.format(fout))
    return (R,L)

def plot_time(fin, fout):
    plt.clf()
    with open(fin) as f:
        data = f.read()
    # train, test information
    A_train = re.findall('A\t(.*)\t', data)[0]
    A_test = re.findall('A\t.*\t(.*)', data)[0]
    B_train = re.findall('B\t(.*)\t', data)[0]
    B_test = re.findall('B\t.*\t(.*)', data)[0]

    # baseline
    A_base_time = re.findall('training_time: (.*)s\n\[A_Baseline\] acc = .*', data)[-1]
    B_base_time = re.findall('training_time: (.*)s\n\[B_Baseline\] acc = .*', data)[-1]
    A, = plt.plot([0], [A_base_time], marker='o', markersize=12, label='A baseline')
    B, = plt.plot([0], [B_base_time], marker='o', markersize=12, label='B baseline')
    # Low rank
    R = set([int(x) for x in re.findall('\[.*_(\d*)_rank1_.*\]', data)])
    L = set([int(x) for x in re.findall('\[.*_(\d*)_Layer\]', data)])
    PLOT = []
    for r in R:
        x, y = [0], [B_base_time]
        for l in L:
            time = re.findall('training_time: (.*)s\n\[B_{0}_rank1_{1}_Layer\]'.format(r,l), data)[-1]
            x += [l]
            y += [time]
        PLOT += [plt.plot(x, y, marker='.', markersize=5, label='{0} Rank1'.format(r))]
    PLOT += [A, B]
    plt.title('Training Time')
    plt.legend()
    plt.xlabel('#transfered layers')
    plt.ylabel('time (s)')
    plt.savefig('{0}_time.png'.format(fout))
    return (R,L)

def plot_loss(fname, run):
    plt.clf()
    X = pd.read_pickle(fname)
    EPOCH = range(len(X))
    tloss, vloss = [X[i]['train_loss'] for i in EPOCH], [X[i]['valid_loss'] for i in EPOCH]
    plt.plot(EPOCH, tloss, label='train loss')
    plt.plot(EPOCH, vloss, label='valid loss')
    plt.title(run)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('{0}.png'.format(run))

def plot_all_loss(R, L):
    plt.clf()
    # Baseline
    plot_loss('A_Baseline.pkl', 'A_baseline')
    # other
    for r in R:
        for l in L:
            plot_loss('B_{0}_rank1_{1}_Layer.pkl'.format(r,l), 'B_{0}R_{1}L'.format(r,l))
    
opts = parse_arg()
fin, fout = opts.inFile, opts.outFile
plot_acc(fin, fout)
R,L = plot_time(fin, fout)
plot_all_loss(R, L)