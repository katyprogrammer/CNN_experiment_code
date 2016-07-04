import sys
import subprocess
import multiprocessing as mp
from Queue import Queue
from threading import Thread

def A():
    subprocess.call('python CNN.py -r A -d A.pkl -e 100 > A.txt', shell=True)
def B():
    subprocess.call('python CNN.py -r B -d B.pkl -e 100 > B.txt', shell=True)

runA, runB = mp.Process(target=A), mp.Process(target=B)
runA.start()
runA.join()
runB.start()
runB.join()

def runR(r):
    subprocess.call('python CNN.py -r B -l A.pkl -d B_{0}.pkl -e 100 -R {0} > B_{0}.txt'.format(r), shell=True)
def runRevR(r):
    subprocess.call('python CNN.py -r A -l B.pkl -d A_{0}.pkl -e 100 -R {0} > A_{0}.txt'.format(r), shell=True)

runs = []
tn = 0
# all layers
R = [1,5,10,50,100,200,256]
# conv only
R = [1,5,10,20,32]
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
for r in R:
    AB = mp.Process(target=runR, args=(r,))
    AB.start()
    AB.join()
    BA = mp.Process(target=runRevR, args=(r,))
    BA.start()
    BA.join()