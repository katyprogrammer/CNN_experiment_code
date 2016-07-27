import sys
import subprocess
import multiprocessing as mp

def A():
    subprocess.call('python CNN.py -r A -d A.pkl -e 100 -t 1 2 3 > A.txt', shell=True)
def B():
    subprocess.call('python CNN.py -r B -d B.pkl -e 100 -t 1 2 3 > B.txt', shell=True)

runA, runB = mp.Process(target=A), mp.Process(target=B)
runA.start()
runA.join()
runB.start()
runB.join()


def runR(r, i):
    # LAYERS = ['conv_1','maxpool_1','conv_2','maxpool_2','dropout_1','dense_1','dropout_2','dense_output']
    subprocess.call('THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python CNN.py -r B -l A.pkl -d B_{0}.pkl -e 30 -R {0} -t conv_1 > B_{0}_{1}.txt'.format(r, i), shell=True)
    subprocess.call('THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python CNN.py -r B -l A.pkl -d B_{0}.pkl -e 30 -R {0} -t conv_1 maxpool_1 > B_{0}_{1}.txt'.format(r, i), shell=True)
    subprocess.call('THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python CNN.py -r B -l A.pkl -d B_{0}.pkl -e 30 -R {0} -t conv_1 maxpool_1 conv_2 > B_{0}_{1}.txt'.format(r, i), shell=True)
    subprocess.call('THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python CNN.py -r B -l A.pkl -d B_{0}.pkl -e 30 -R {0} -t conv_1 maxpool_1 conv_2 maxpool_2 > B_{0}_{1}.txt'.format(r, i), shell=True)
    subprocess.call('THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python CNN.py -r B -l A.pkl -d B_{0}.pkl -e 30 -R {0} -t conv_1 maxpool_1 conv_2 maxpool_2 dropout_1 > B_{0}_{1}.txt'.format(r, i), shell=True)
    subprocess.call('THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python CNN.py -r B -l A.pkl -d B_{0}.pkl -e 30 -R {0} -t conv_1 maxpool_1 conv_2 maxpool_2 dropout_1 dense_1 > B_{0}_{1}.txt'.format(r, i), shell=True)
    subprocess.call('THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python CNN.py -r B -l A.pkl -d B_{0}.pkl -e 30 -R {0} -t conv_1 maxpool_1 conv_2 maxpool_2 dropout_1 dense_1 dropout_2 > B_{0}_{1}.txt'.format(r, i), shell=True)
    subprocess.call('THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python CNN.py -r B -l A.pkl -d B_{0}.pkl -e 30 -R {0} -t conv_1 maxpool_1 conv_2 maxpool_2 dropout_1 dense_1 dropout_2 dense_output > B_{0}_{1}.txt'.format(r, i), shell=True)
def runRevR(r):
    # subprocess.call('THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python CNN.py -r A -l B.pkl -d A_{0}.pkl -e 30 -R {0} > A_{0}.txt'.format(r), shell=True)
    subprocess.call('THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python CNN.py -r A -l B.pkl -d A_{0}.pkl -e 30 -R {0} -t conv_1 > A_{0}_{1}.txt'.format(r, i), shell=True)
    subprocess.call('THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python CNN.py -r A -l B.pkl -d A_{0}.pkl -e 30 -R {0} -t conv_1 maxpool_1 > A_{0}_{1}.txt'.format(r, i), shell=True)
    subprocess.call('THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python CNN.py -r A -l B.pkl -d A_{0}.pkl -e 30 -R {0} -t conv_1 maxpool_1 conv_2 > A_{0}_{1}.txt'.format(r, i), shell=True)
    subprocess.call('THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python CNN.py -r A -l B.pkl -d A_{0}.pkl -e 30 -R {0} -t conv_1 maxpool_1 conv_2 maxpool_2 > A_{0}_{1}.txt'.format(r, i), shell=True)
    subprocess.call('THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python CNN.py -r A -l B.pkl -d A_{0}.pkl -e 30 -R {0} -t conv_1 maxpool_1 conv_2 maxpool_2 dropout_1 > A_{0}_{1}.txt'.format(r, i), shell=True)
    subprocess.call('THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python CNN.py -r A -l B.pkl -d A_{0}.pkl -e 30 -R {0} -t conv_1 maxpool_1 conv_2 maxpool_2 dropout_1 dense_1 > A_{0}_{1}.txt'.format(r, i), shell=True)
    subprocess.call('THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python CNN.py -r A -l B.pkl -d A_{0}.pkl -e 30 -R {0} -t conv_1 maxpool_1 conv_2 maxpool_2 dropout_1 dense_1 dropout_2 > A_{0}_{1}.txt'.format(r, i), shell=True)
    subprocess.call('THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python CNN.py -r A -l B.pkl -d A_{0}.pkl -e 30 -R {0} -t conv_1 maxpool_1 conv_2 maxpool_2 dropout_1 dense_1 dropout_2 dense_output > A_{0}_{1}.txt'.format(r, i), shell=True)
runs = []
tn = 0
# all layers
R = [1,5,10,50,100,200,256]
# conv only
# R = [1,5,10,20,32]
# parallel
# while len(R) > 0:
#     if tn < 4:
#         r = R.pop(0)
#         runs.append(mp.Process(target=runRevR, args=(r,)))
#         runs[-1].start()
#         tn += 1
#     else:
#         for run in runs:
#             run.join()
#         tn = 0

tn = 0
for i in range(5):
    for r in R:
        AB = mp.Process(target=runR, args=(r, i))
        AB.start()
        AB.join()
        BA = mp.Process(target=runRevR, args=(r, i))
        BA.start()
        BA.join()
