# coding=utf-8
from emolga.dataset.build_dataset import deserialize_from_file, serialize_to_file
import string
import random
import sys

random.seed(19920206)
word2idx  = dict()
wordfreq  = dict()
word2idx['<eol>'] = 0
word2idx['<unk>'] = 1
word2freq = dict()


def mark(line):
    tmp_line = ''
    for c in line:
        if c in string.punctuation:
            if c is not "'":
                tmp_line += ' ' + c + ' '
            else:
                tmp_line += ' ' + c
        else:
            tmp_line += c
    tmp_line = tmp_line.lower()
    words = [w for w in tmp_line.split() if len(w) > 0]
    for w in words:
        if w not in word2freq:
            word2freq[w]  = 1
        else:
            word2freq[w] += 1
    return words


fline     = open('./dataset/cornell_movie/movie_lines.txt', 'r')
sets      = [w.split('+++$+++') for w in fline.read().split('\n')]
lines     = {w[0].strip(): mark(w[-1].strip()) for w in sets}
#
# for w in lines:
#     if len(lines[w]) == 0:
#         print w

fline.close()
print 'read lines ok'
fconv     = open('./dataset/cornell_movie/movie_conversations.txt', 'r')

turns     = []
convs     = fconv.readline()
while convs:
    turn   = eval(convs.split('+++$+++')[-1].strip())
    turns += zip(turn[:-1], turn[1:])
    convs  = fconv.readline()

pairs     = [(lines[a], lines[b]) for a, b in turns
             if len(lines[a]) > 0 and len(lines[b]) > 0]

# shuffle!
random.shuffle(pairs)

word2freq = sorted(word2freq.items(), key=lambda a: a[1], reverse=True)
for at, w in enumerate(word2freq):
    word2idx[w[0]] = at + 2

idx2word  = {k: v for v, k in word2idx.items()}
print idx2word[1], idx2word[2]

Lmax     = len(idx2word)
# for i in xrange(Lmax):
#     print idx2word[i]
print 'read dataset ok.'
print Lmax
print pairs[0]


def build_data(data):
    instance = dict(text=[], summary=[], source=[], target=[], target_c=[])
    print len(data)
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

    print instance['source'][4000]
    print instance['target'][4000]
    print instance['target_c'][4000]
    return instance


train_set = build_data(pairs[10000:])
test_set  = build_data(pairs[:10000])
serialize_to_file([train_set, test_set, idx2word, word2idx], './dataset/movie_dialogue_data.pkl')
