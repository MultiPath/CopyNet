# coding=utf-8
import chardet
import sys
import numpy as np
import jieba as jb
from emolga.dataset.build_dataset import deserialize_from_file, serialize_to_file

word2idx = dict()
wordfreq = dict()
word2idx['<eol>'] = 0
word2idx['<unk>'] = 1

segment  = False # True

# training set
pairs = []
f     = open('./dataset/LCSTS/PART_I/PART_full.txt', 'r')
line  = f.readline().strip()
at    = 2
lines = 0
while line:
    if line == '<summary>':
        summary = f.readline().strip().decode('utf-8')
        if segment:
            summary = [w for w in jb.cut(summary)]

        for w in summary:
            if w not in wordfreq:
                wordfreq[w]  = 1
            else:
                wordfreq[w] += 1
            # if w not in word2idx:
            #     word2idx[w] = at
            #     at         += 1

        f.readline()
        f.readline()
        text    = f.readline().strip().decode('utf-8')
        if segment:
            text = [w for w in jb.cut(text)]
        for w in text:
            if w not in wordfreq:
                wordfreq[w]  = 1
            else:
                wordfreq[w] += 1
            # if w not in word2idx:
            #     word2idx[w] = at
            #     at         += 1

        pair    = (text, summary)
        pairs.append(pair)
        lines  += 1
        if lines % 20000 == 0:
            print lines
    line = f.readline().strip()

# testing set
tests = []
f     = open('./dataset/LCSTS/PART_II/PART_II.txt', 'r')
line  = f.readline().strip()
lines = 0
while line:
    if line == '<summary>':
        summary = f.readline().strip().decode('utf-8')
        if segment:
            summary = [w for w in jb.cut(summary)]

        for w in summary:
            if w not in wordfreq:
                wordfreq[w]  = 1
            else:
                wordfreq[w] += 1
            # if w not in word2idx:
            #     word2idx[w] = at
            #     at         += 1

        f.readline()
        f.readline()
        text    = f.readline().strip().decode('utf-8')
        if segment:
            text = [w for w in jb.cut(text)]
        for w in text:
            if w not in wordfreq:
                wordfreq[w]  = 1
            else:
                wordfreq[w] += 1
            # if w not in word2idx:
            #     word2idx[w] = at
            #     at         += 1

        pair    = (text, summary)
        tests.append(pair)
        lines  += 1
        if lines % 20000 == 0:
            print lines
    line = f.readline().strip()

print len(pairs), len(tests)

# sort the vocabulary
wordfreq = sorted(wordfreq.items(), key=lambda a:a[1], reverse=True)
for w in wordfreq:
    word2idx[w[0]] = at
    at += 1

idx2word = {k: v for v, k in word2idx.items()}
Lmax     = len(idx2word)
print 'read dataset ok.'
print Lmax
for i in xrange(Lmax):
    print idx2word[i].encode('utf-8')

# use character-based model [on]
# use word-based model     [off]


def build_data(data):
    instance = dict(text=[], summary=[], source=[], target=[], target_c=[])
    for pair in data:
        source, target = pair
        A = [word2idx[w] for w in source]
        B = [word2idx[w] for w in target]
        # C = np.asarray([[w == l for w in source] for l in target], dtype='float32')
        C = [0 if w not in source else source.index(w) + Lmax for w in target]

        instance['text']      += [source]
        instance['summary']   += [target]
        instance['source']    += [A]
        instance['target']    += [B]
        # instance['cc_matrix'] += [C]
        instance['target_c'] += [C]

    print instance['target'][5000]
    print instance['target_c'][5000]
    return instance


train_set = build_data(pairs)
test_set  = build_data(tests)
serialize_to_file([train_set, test_set, idx2word, word2idx], './dataset/lcsts_data-char-full.pkl')
