__author__ = 'jiataogu'
from emolga.dataset.build_dataset import deserialize_from_file, serialize_to_file
import numpy.random as n_rng

n_rng.seed(19920206)
# the vocabulary
tmp      = [chr(x) for x in range(48, 58)]  # '1', ... , '9', '0'
voc      = [tmp[a] + tmp[b] + tmp[c]
            for c in xrange(10)
            for b in xrange(10)
            for a in xrange(10)]
word2idx           = {voc[k]: k + 2 for k in xrange(len(voc))}
word2idx['<eol>']  = 0
word2idx['<unk>']  = 1
idx2word           = {word2idx[w]: w for w in word2idx}
voc                = ['<eol>', '<unk>'] + voc

# word2idx['X']      = len(voc)
# idx2word[len(voc)] = 'X'
# voc               += ['X']
#
# word2idx['Y']      = len(voc)
# idx2word[len(voc)] = 'Y'
# voc               += ['Y']
# print word2idx['X'], word2idx['Y']

# load the dataset
Rules, _ = deserialize_from_file('/home/thoma/Work/Dial-DRL/dataset/rules.rnd.n10k.pkl')
num      = 200
repeats  = 100
maxleg   = 15
Lmax     = len(idx2word)
rules    = dict(source=Rules['source'][:num],
                target=Rules['target'][:num])


def ftr(v):
    if v < 10:
        return '00' + str(v)
    elif v < 100:
        return '0' + str(v)
    else:
        return str(v)


def build_instance():
    instance = dict(x=[], y=[], source=[], target=[], target_c=[], rule_id=[], rule=[])
    for k in xrange(num):
        source = rules['source'][k]
        target = rules['target'][k]

        for j in xrange(repeats):
            X  = n_rng.randint(1000, size= n_rng.randint(maxleg) + 1)
            Y  = n_rng.randint(1000, size= n_rng.randint(maxleg) + 1)
            S  = []
            T  = []
            for w in source:
                if w is 'X':
                    S += [ftr(v) for v in X]
                elif w is 'Y':
                    S += [ftr(v) for v in Y]
                else:
                    S += [w]

            for w in target:
                if w is 'X':
                    T += [ftr(v) for v in X]
                elif w is 'Y':
                    T += [ftr(v) for v in Y]
                else:
                    T += [w]

            A  = [word2idx[w] for w in S]
            B  = [word2idx[w] for w in T]
            C  = [0 if w not in S else S.index(w) + Lmax for w in T]

            instance['x']        += [S]
            instance['y']        += [T]
            instance['source']   += [A]
            instance['target']   += [B]
            instance['target_c'] += [C]

            instance['rule_id']  += [k]
            instance['rule']     += [' '.join(source) + ' -> ' + ' '.join(target)]

    return instance

train_set = build_instance()
print 'build ok.'
test_set  = build_instance()
print 'build ok.'

serialize_to_file([train_set, test_set, idx2word, word2idx], '/home/thoma/Work/Dial-DRL/dataset/synthetic_data_c.pkl')
# serialize_to_file([train_set, test_set], '/home/thoma/Work/Dial-DRL/dataset/synthetic_data.pkl')
