import optparse
import matplotlib.pyplot as plt
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
    epoch = re.findall(EPOCH, data)
    tloss = re.findall(TLOSS, data)
    vloss = re.findall(VLOSS, data)
    vacc = re.findall(VACC, data)
    plt.plot(epoch, tloss, marker='.', markersize=1, label='training loss')
    plt.plot(epoch, vloss, marker='.', markersize=1, label='validation loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('loss_{0}.png'.format(fout))
    plt.clf()
    plt.plot(epoch, vacc, marker='.', markersize=1, label='validation accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('val_acc_{0}.png'.format(fout))

opts = parse_arg()
plot(opts.fin, opts.fout)