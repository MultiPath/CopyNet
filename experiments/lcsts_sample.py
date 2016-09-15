"""
This is the implementation of Copy-NET
We start from the basic Seq2seq framework for a auto-encoder.
"""
import logging
import time
import numpy as np
import sys
import copy

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from experiments.config import setup_lcsts
from emolga.utils.generic_utils import *
from emolga.models.covc_encdec import NRM
from emolga.models.encdec import NRM as NRM0
from emolga.dataset.build_dataset import deserialize_from_file
from collections import OrderedDict
from fuel import datasets
from fuel import transformers
from fuel import schemes

setup = setup_lcsts


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
tmark   = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
config  = setup()   # load settings.
for w in config:
    print '{0}={1}'.format(w, config[w])

logger  = init_logging(config['path_log'] + '/experiments.CopyLCSTS.id={}.log'.format(tmark))
n_rng   = np.random.RandomState(config['seed'])
np.random.seed(config['seed'])
rng     = RandomStreams(n_rng.randint(2 ** 30))
logger.info('Start!')

train_set, test_set, idx2word, word2idx = deserialize_from_file(config['dataset'])
if config['voc_size'] == -1:   # not use unk
    config['enc_voc_size'] = max(zip(*word2idx.items())[1]) + 1
    config['dec_voc_size'] = config['enc_voc_size']
else:
    config['enc_voc_size'] = config['voc_size']
    config['dec_voc_size'] = config['enc_voc_size']

samples = len(train_set['source'])

logger.info('build dataset done. ' +
            'dataset size: {} ||'.format(samples) +
            'vocabulary size = {0}/ batch size = {1}'.format(
        config['dec_voc_size'], config['batch_size']))


def build_data(data):
    # create fuel dataset.
    dataset     = datasets.IndexableDataset(indexables=OrderedDict([('source', data['source']),
                                                                    ('target', data['target']),
                                                                    ('target_c', data['target_c']),
                                                                    ]))
    dataset.example_iteration_scheme \
                = schemes.ShuffledExampleScheme(dataset.num_examples)
    return dataset


def unk_filter(data):
    if config['voc_size'] == -1:
        return copy.copy(data)
    else:
        mask = (np.less(data, config['voc_size'])).astype(dtype='int32')
        data = copy.copy(data * mask + (1 - mask))
        return data


train_data_plain  = zip(*(train_set['source'], train_set['target']))
test_data_plain   = zip(*(test_set['source'],  test_set['target']))
train_size        = len(train_data_plain)
test_size         = len(test_data_plain)
tr_idx            = n_rng.permutation(train_size)[:2000].tolist()
ts_idx            = n_rng.permutation(test_size)[:100].tolist()

logger.info('load the data ok.')

# logger.info('Evaluate Enc-Dec')
# log_gen           = open(config['path_log'] + '/experiments.CopyLCSTS.generate_{}.log'.format(0), 'w')
# config['copynet'] = True
# echo              = 10
# tmark             = '20160224-185023'  # '20160221-171853'  # enc-dec model [no unk]
# agent  = NRM(config, n_rng, rng, mode=config['mode'],
#                   use_attention=True, copynet=config['copynet'], identity=config['identity'])
# agent.build_()
# agent.compile_('display')
# agent.load(config['path_h5'] + '/experiments.CopyLCSTS.id={0}.epoch={1}.pkl'.format(tmark, echo))
# logger.info('generating [testing set] samples')
# for idx in ts_idx:
#     # idx            = int(np.floor(n_rng.rand() * test_size))
#     test_s, test_t = test_data_plain[idx]
#     v              = agent.evaluate_(np.asarray(test_s, dtype='int32'),
#                                      np.asarray(test_t, dtype='int32'),
#                                      idx2word)
#     log_gen.write(v)
#     log_gen.write('*' * 50 + '\n')
# log_gen.close()

logger.info('Evaluate CopyNet')
echo              = 6
tmark             = '20160224-185023'  # '20160221-025049'  # copy-net model [no unk]
log_cp            = open(config['path_logX'] + '/experiments.copy_{0}_{1}.log'.format(tmark, echo), 'w')
config['copynet'] = True
agent  = NRM(config, n_rng, rng, mode=config['mode'],
                  use_attention=True, copynet=config['copynet'], identity=config['identity'])
agent.build_()
agent.compile_('display')
agent.load(config['path_h5'] + '/experiments.CopyLCSTS.id={0}.epoch={1}.pkl'.format(tmark, echo))
logger.info('generating [testing set] samples')
for idx in ts_idx:
    # idx            = int(np.floor(n_rng.rand() * test_size))
    test_s, test_t = test_data_plain[idx]
    v              = agent.evaluate_(np.asarray(test_s, dtype='int32'),
                                     np.asarray(test_t, dtype='int32'),
                                     idx2word, np.asarray(unk_filter(test_s), dtype='int32'))
    log_cp.write(v)
    log_cp.write('*' * 50 + '\n')
log_cp.close()
logger.info('Complete!')