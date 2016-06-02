from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import HashingVectorizer
import cPickle
from os.path import join
from pprint import pprint


newsgroups = fetch_20newsgroups(subset='all')
hasher = HashingVectorizer(n_features=100)
vectors = hasher.fit_transform(newsgroups.data)

target = newsgroups.target
l = len(target)
for i in range(20):
    x = []
    for j in range(l):
        if target[j]==i:
            x.append(vectors[j])
    cPickle.dump(x, open(join('train','{0}.pkl'.format(i)), 'w+'))