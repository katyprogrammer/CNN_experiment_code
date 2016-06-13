import optparse
import matplotlib.pyplot as plt
import pandas as pd
import re

EPOCH = 'Epoch (\d*).*\n.*training loss:\t\t.*\n.*validation loss:\t\t.*\n.*validation accuracy:\t\t.* %'
TLOSS = 'Epoch \d*.*\n.*training loss:\t\t(.*)\n.*validation loss:\t\t.*\n.*validation accuracy:\t\t.* %'
VLOSS = 'Epoch \d*.*\n.*training loss:\t\t.*\n.*validation loss:\t\t(.*)\n.*validation accuracy:\t\t.* %'
VACC = 'Epoch \d*.*\n.*training loss:\t\t.*\n.*validation loss:\t\t.*\n.*validation accuracy:\t\t(.*) %'


def parse_arg():
    parser = optparse.OptionParser('usage%prog [-i infile]')
    parser.add_option('-i', dest='fin')
    parser.add_option('-o', dest='fout')
    (options, args) = parser.parse_args()
    return options

def plot(fin,fout):
    with open(fin, 'r') as f:
        data = f.read()
    epoch = [float(x) for x in re.findall(EPOCH, data)]
    tloss = [float(x) for x in re.findall(TLOSS, data)]
    vloss = [float(x) for x in re.findall(VLOSS, data)]
    vacc = [float(x) for x in re.findall(VACC, data)]
    df = pd.DataFrame({'training loss':tloss, 'validation loss':vloss}, index=epoch)
    df.plot()
    plt.xlabel('epoch')
    plt.savefig('loss_{0}.png'.format(fout))
    plt.clf()
    df = pd.DataFrame({'validation accuracy':vacc}, index=epoch)
    df.plot()
    plt.xlabel('epoch')
    plt.savefig('val_acc_{0}.png'.format(fout))

opts = parse_arg()
plot(opts.fin, opts.fout)