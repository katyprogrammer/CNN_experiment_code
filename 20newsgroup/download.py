from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import HashingVectorizer
import cPickle
from os.path import join, exists
import os
from pprint import pprint

if not exists('train'):
    os.makedirs('train')


newsgroups = fetch_20newsgroups(subset='all')
# experiment on reasonable input
hasher = HashingVectorizer(n_features=10000)
vectors = hasher.fit_transform(newsgroups.data)
    
target = newsgroups.target
l = len(target)
for i in range(20):
    x = []
    for j in range(l):
        if target[j]==i:
            x.append(vectors[j])
    cPickle.dump(x, open(join('train','{0}.pkl'.format(i)), 'w+'))
