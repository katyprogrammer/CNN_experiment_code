import multiprocessing as mp
import subprocess
import os

def run(r):
    os.system('python 20newsgroup.py -r B -l A.pkl -d B_{0}.pkl -e 10000 -R 1 > B_{0}.txt'.format(r))

pool = mp.Pool(processes=15)
for r in [1,2,3,4,5,6,7,8,9,10,20,50,100,200,500]:
    pool.apply_async(run, args=[r])