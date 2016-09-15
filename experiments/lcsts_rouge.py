"""
Evaluation using ROUGE for LCSTS dataset.
"""
# load the testing set.
from emolga.dataset.build_dataset import deserialize_from_file, serialize_to_file
import jieba as jb
import logging
import copy
from pyrouge import Rouge155
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from experiments.config import setup_lcsts, setup_weibo, setup_syn
from emolga.utils.generic_utils import *
from emolga.models.covc_encdec import NRM
from emolga.models.encdec import NRM as NRM0
from emolga.dataset.build_dataset import deserialize_from_file
from collections import OrderedDict
from fuel import datasets
from fuel import transformers
from fuel import schemes
from pprint import pprint
setup = setup_lcsts


def build_evaluation(train_set, segment):
    _, _, idx2word, word2idx = deserialize_from_file(train_set)
    pairs   = []
    f       = open('./dataset/LCSTS/PART_III/PART_III.txt', 'r')
    line    = f.readline().strip()
    lines   = 0
    segment = segment
    while line:
        if '<human_label>' in line:
            score   = int(line[13])
            if score >= 3:
                f.readline()
                summary = f.readline().strip().decode('utf-8')
                if segment:
                    summary = [w for w in jb.cut(summary)]
                target  = []
                for w in summary:
                    if w not in word2idx:
                        word2idx[w] = len(word2idx)
                        idx2word[len(idx2word)] = w
                    target += [word2idx[w]]

                f.readline()
                f.readline()
                text    = f.readline().strip().decode('utf-8')
                if segment:
                    text = [w for w in jb.cut(text)]
                source  = []
                for w in text:
                    if w not in word2idx:
                        word2idx[w] = len(word2idx)
                        idx2word[len(idx2word)] = w
                    source += [word2idx[w]]

                pair    = (text, summary, score, source, target)
                pairs.append(pair)
                lines  += 1
                if lines % 1000 == 0:
                    print lines
        line = f.readline().strip()
    print 'lines={}'.format(len(pairs))
    return pairs, word2idx, idx2word

# words, wwi, wiw = build_evaluation('./dataset/lcsts_data-word-full.pkl', True)
# chars, cwi, ciw = build_evaluation('./dataset/lcsts_data-char-full.pkl', False)
#
# serialize_to_file([words, chars, [wwi, wiw], [cwi, ciw]], './dataset/lcsts_evaluate_data.pkl')


def init_logging(logfile):
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(module)s: %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S'   )
    fh = logging.FileHandler(logfile)
    # ch = logging.StreamHandler()

    fh.setFormatter(formatter)
    # ch.setFormatter(formatter)
    # fh.setLevel(logging.INFO)
    # ch.setLevel(logging.INFO)
    # logging.getLogger().addHandler(ch)
    logging.getLogger().addHandler(fh)
    logging.getLogger().setLevel(logging.INFO)

    return logging


# prepare logging.
config  = setup()   # load settings.
for w in config:
    print '{0}={1}'.format(w, config[w])
tmark   = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
logger  = init_logging(config['path_log'] + '/experiments.LCSTS.Eval.id={}.log'.format(tmark))
n_rng   = np.random.RandomState(config['seed'])
np.random.seed(config['seed'])
rng     = RandomStreams(n_rng.randint(2 ** 30))
logger.info('Start!')

segment = config['segment']
word_set, char_set, word_voc, char_voc = deserialize_from_file('./dataset/lcsts_evaluate_data.pkl')

if segment:
    eval_set           = word_set
    word2idx, idx2word = word_voc
else:
    eval_set           = char_set
    word2idx, idx2word = char_voc

if config['voc_size'] == -1:   # not use unk
    config['enc_voc_size'] = len(word2idx)
    config['dec_voc_size'] = config['enc_voc_size']
else:
    config['enc_voc_size'] = config['voc_size']
    config['dec_voc_size'] = config['enc_voc_size']

samples  = len(eval_set)
logger.info('build dataset done. ' +
            'dataset size: {} ||'.format(samples) +
            'vocabulary size = {0}/ batch size = {1}'.format(
        config['dec_voc_size'], config['batch_size']))
logger.info('load the data ok.')

# build the agent
if config['copynet']:
    agent  = NRM(config, n_rng, rng, mode=config['mode'],
                 use_attention=True, copynet=config['copynet'], identity=config['identity'])
else:
    agent  = NRM0(config, n_rng, rng, mode=config['mode'],
                  use_attention=True, copynet=config['copynet'], identity=config['identity'])

agent.build_()
agent.compile_('display')
print 'compile ok.'

# load the model
agent.load(config['trained_model'])


def unk_filter(data):
    if config['voc_size'] == -1:
        return copy.copy(data)
    else:
        mask = (np.less(data, config['voc_size'])).astype(dtype='int32')
        data = copy.copy(data * mask + (1 - mask))
        return data

rouge    = Rouge155(n_words=40)
evalsets = {'rouge_1_f_score': 'R1',
            'rouge_2_f_score': 'R2',
            'rouge_3_f_score': 'R3',
            'rouge_4_f_score': 'R4',
            'rouge_l_f_score': 'RL',
            'rouge_su4_f_score': 'RSU4'}
scores = dict()
for id, sample in enumerate(eval_set):
    text, summary, score, source, target = sample
    v              = agent.evaluate_(np.asarray(source, dtype='int32'),
                                     np.asarray(target, dtype='int32'),
                                     idx2word,
                                     np.asarray(unk_filter(source), dtype='int32')).decode('utf-8').split('\n')

    print 'ID = {} ||'.format(id) + '*' * 50
    ref   = ' '.join(['t{}'.format(char_voc[0][u]) for u in ''.join([w for w in v[2][9:].split()])])
    sym   = ' '.join(['t{}'.format(char_voc[0][u]) for u in ''.join([w for w in v[3][9:].split()])])

    sssss = rouge.score_summary(sym, {'A': ref})

    for si in sssss:
        if si not in scores:
            scores[si]  = sssss[si]
        else:
            scores[si] += sssss[si]

    for e in evalsets:
        print '{0}: {1}'.format(evalsets[e], scores[e] / (id + 1)),
    print './.'

# average
for si in scores:
    scores[si] /= float(len(eval_set))
