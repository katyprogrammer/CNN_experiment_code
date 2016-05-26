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
    B_base_raw_acc = re.findall('training_time: .*\n\[B_B_Raw\] acc = (.*)', data)[-1]
    # A_acc, = plt.plot([0], [A_base_acc], marker='o', markersize=12, label='A baseline')
    B_acc, = plt.plot([0], [B_base_acc], marker='o', markersize=12, label='B baseline')
    B_raw_acc, = plt.plot([0], [B_base_raw_acc], marker='*', markersize=12, label='B (all layer of A)')
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
        plt.plot(x, y, marker='.', markersize=5, label='{0} Rank1'.format(r))
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
    # baseline
    A_base_time = re.findall('training_time: (.*)s\n\[A_Baseline\] acc = .*', data)[-1]
    B_base_time = re.findall('training_time: (.*)s\n\[B_Baseline\] acc = .*', data)[-1]
    B_base_raw_time = re.findall('training_time: (.*)s\n\[B_B_Raw\] acc = .*', data)[-1]
    # A, = plt.plot([0], [A_base_time], marker='o', markersize=12, label='A baseline')
    B, = plt.plot([0], [B_base_time], marker='o', markersize=12, label='B baseline')
    B_raw, = plt.plot([0], [B_base_raw_time], marker='*', markersize=12, label='B (all layer of A)')
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
        plt.plot(x, y, marker='.', markersize=5, label='{0} Rank1'.format(r))
    plt.title('Training Time')
    plt.legend()
    plt.xlabel('#transfered layers')
    plt.ylabel('time (s)')
    plt.savefig('{0}_time.png'.format(fout))
    return (R,L)

def get_epoch(fname):
    global ACC
    X = pd.read_pickle(fname)
    EPOCH = range(len(X))
    for i in EPOCH:
        if float(X[i]['valid_accuracy']) > ACC:
            return i

def plot_epoch_tgt(fname, R, L):
    plt.clf()
    # Baseline
    A, B, B_raw = get_epoch('A_Baseline.pkl'), get_epoch('B_Baseline.pkl'), get_epoch('B_B_Raw.pkl')
    # plt.plot([0], [A], label='A Baseline', marker='o', markersize=12)
    plt.plot([0], [B], label='B Baseline', marker='o', markersize=12)
    plt.plot([0], [B_raw], label='B (all layers of A)', marker='*', markersize=12)
    # other
    for r in R:
        EPOCH = []
        for l in L:
            EPOCH += [get_epoch('B_{0}_rank1_{1}_Layer.pkl'.format(r,l))]
        plt.plot(list(L), EPOCH, label='{0} Rank1 approximate'.format(r), marker='.', markersize=5)
    plt.legend()
    plt.title('# epoch to acc > 0.9')
    plt.xlabel('# transfered layer')
    plt.ylabel('# epoch')
    plt.savefig(fname)
            

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

def plot_acc_epoch(fname, run, baseA=None, baseB=None, baseBR=None):
    plt.clf()
    X = pd.read_pickle(fname)
    EPOCH = range(len(X))
    vacc = [X[i]['valid_accuracy'] for i in EPOCH]
    plt.plot(EPOCH, vacc, label='valid accuracy')
    if baseA is not None:
        plt.plot(range(len(baseA)), baseA, label='A baseline', linestyle='-.')
    if baseB is not None:
        plt.plot(range(len(baseB)), baseB, label='B baseline', linestyle='-.')
    if baseBR is not None:
        plt.plot(range(len(baseBR)), baseBR, label='B (all layers of A)', linestyle='-.')
    plt.title(run)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig('{0}_acc.png'.format(run))
    return vacc

def plot_all(R, L):
    plt.clf()
    # Baseline
    # X, XN = 'A_Baseline.pkl', 'A_baseline'
    # plot_loss(X, XN)
    # A = plot_acc_epoch(X, XN)
    X, XN = 'B_Baseline.pkl', 'B_baseline'
    # plot_loss(X, XN)
    B = plot_acc_epoch(X, XN)
    X, XN = 'B_B_Raw.pkl', 'B_R-1'
    B_raw = plot_acc_epoch(X, XN)
    # other
    for r in R:
        for l in L:
            X, XN = 'B_{0}_rank1_{1}_Layer.pkl'.format(r,l), 'B_{0}R_{1}L'.format(r,l)
            # plot_loss(X, XN)
            plot_acc_epoch(X, XN, baseA=None, baseB=B, baseBR=B_raw)

ACC = 0.9
opts = parse_arg()
fin, fout = opts.inFile, opts.outFile
plot_acc(fin, fout)
R,L = plot_time(fin, fout)
plot_all(R, L)
plot_epoch_tgt('EpochToTgt.png', R, L)