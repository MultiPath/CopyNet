"""
Preprocess the bAbI datset.
"""
import logging
import os
import sys
import numpy.random as n_rng
from emolga.dataset.build_dataset import serialize_to_file

data_path = './dataset/bAbI/en-10k/'
data = []
n_rng.seed(19920206)

for p, folders, docs in os.walk(data_path):
    for doc in docs:
        with open(os.path.join(p, doc)) as f:
            l = f.readline()
            while l:
                l = l.strip().lower()
                l = l[l.find(' ') + 1:]
                if len(l.split('\t')) == 1:
                    data += [l[:-1].split()]
                l = f.readline()

idx2word = dict(enumerate(set([w for l in data for w in l]), 1))
word2idx = {v: k for k, v in idx2word.items()}

persons  = [1, 6, 24, 37, 38, 47, 60, 61, 73, 74, 90, 94, 107, 110, 114]
colors   = [3, 20, 34, 48, 99, 121]
shapes   = [11, 15, 27, 99]


def repeat_name(l):
    ll = []
    for word in l:
        if word2idx[word] in persons:
            k = n_rng.randint(5) + 1
            ll += [idx2word[persons[i]] for i in n_rng.randint(len(persons), size=k).tolist()]
        elif word2idx[word] in colors:
            k = n_rng.randint(5) + 1
            ll += [idx2word[colors[i]] for i in n_rng.randint(len(colors), size=k).tolist()]
        elif word2idx[word] in shapes:
            k = n_rng.randint(5) + 1
            ll += [idx2word[shapes[i]] for i in n_rng.randint(len(shapes), size=k).tolist()]
        else:
            ll += [word]
    return ll

data_rep = [repeat_name(l) for l in data]
origin   = [[word2idx[w] for w in l] for l in data_rep]

def replace(word):
    if word2idx[word] in [1, 6, 24, 37, 38, 47, 60, 61, 73, 74, 90, 94, 107, 110, 114]:
        return '<person>'
    elif word2idx[word] in [3, 20, 34, 48, 99, 121]:
        return '<color>'
    elif word2idx[word] in [11, 15, 27, 99]:
        return '<shape>'
    else:
        return word

# prepare the vocabulary
data_clean   = [[replace(w) for w in l] for l in data_rep]
idx2word2    = dict(enumerate(set([w for l in data_clean for w in l]), 1))
idx2word2[0] = '<eol>'
word2idx2    = {v: k for k, v in idx2word2.items()}
Lmax         = len(idx2word2)

for k in xrange(len(idx2word2)):
    print k, '\t', idx2word2[k]
print 'Max: {}'.format(Lmax)

serialize_to_file([idx2word2, word2idx2, idx2word, word2idx], './dataset/bAbI/voc-b.pkl')

# get ready for the dataset.
source = [[word2idx2[w] for w in l] for l in data_clean]
target = [[word2idx2[w] if w not in ['<person>', '<color>', '<shape>']
           else it + Lmax
           for it, w in enumerate(l)] for l in data_clean]


def print_str(data):
    for d in data:
        print ' '.join(str(w) for w in d)


print_str(data[10000: 10005])
print_str(data_rep[10000: 10005])
print_str(data_clean[10000: 10005])
print_str(source[10000: 10005])
print_str(target[10000: 10005])

serialize_to_file([source, target, origin], './dataset/bAbI/dataset-b.pkl')
