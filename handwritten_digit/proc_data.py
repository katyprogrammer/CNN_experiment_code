import pandas as pd
import os

def split_by_digit(fname):
    if not os.path.exists('train'):
        os.makedirs('train')
    df = pd.read_csv(fname)
    for digit in range(10):
        d = df[df['label']==digit]
        d.to_csv(os.path.join('train','{0}.csv'.format(digit)))

split_by_digit('input/MNIST.csv')