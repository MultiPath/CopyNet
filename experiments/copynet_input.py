# coding=utf-8
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


def unk_filter(data):
    if config['voc_size'] == -1:
        return copy.copy(data)
    else:
        mask = (np.less(data, config['voc_size'])).astype(dtype='int32')
        data = copy.copy(data * mask + (1 - mask))
        return data

source = '临近 岁末 ， 新 基金 发行 步入 旺季 ， 11 月份 以来 单周 新基 ' +  \
         '发行 数 始终保持 35 只 以上 的 高位 ， 仅 11 月 25 日 一天 ， ' + \
         '就 有 12 只 基金 同时 发售 。 国内 首只 公募 对冲 混合型 基金 — 嘉实 绝对 收益 策略 ' + \
         '定期 混合 基金 自 发行 首日 便 备受 各界 青睐 ， 每日 认购 均 能 达到 上 亿'
target = '首只 公募 对冲 基金 每日 吸金 上 亿'

test_s = [word2idx[w.decode('utf-8')] for w in source.split()]
test_t = [word2idx[w.decode('utf-8')] for w in target.split()]

logger.info('load the data ok.')

logger.info('Evaluate CopyNet')
echo              = 9
tmark             = '20160226-164053'  # '20160221-025049'  # copy-net model [no unk]
config['copynet'] = True
agent  = NRM(config, n_rng, rng, mode=config['mode'],
                  use_attention=True, copynet=config['copynet'], identity=config['identity'])
agent.build_()
agent.compile_('display')
agent.load(config['path_h5'] + '/experiments.CopyLCSTS.id={0}.epoch={1}.pkl'.format(tmark, echo))
logger.info('generating [testing set] samples')

v      = agent.evaluate_(np.asarray(test_s, dtype='int32'),
                         np.asarray(test_t, dtype='int32'),
                         idx2word, np.asarray(unk_filter(test_s), dtype='int32'))
logger.info('Complete!')