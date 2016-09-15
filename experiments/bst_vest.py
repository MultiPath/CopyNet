# coding=utf-8
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
from experiments.config import setup_lcsts, setup_weibo, setup_syn, setup_bst
from emolga.utils.generic_utils import *
from emolga.models.covc_encdec import NRM
from emolga.models.encdec import NRM as NRM0
from emolga.dataset.build_dataset import deserialize_from_file
from collections import OrderedDict
from fuel import datasets
from fuel import transformers
from fuel import schemes

# setup = setup_lcsts
# setup = setup_syn
setup = setup_bst


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
    config['enc_voc_size'] = len(word2idx)
    config['dec_voc_size'] = config['enc_voc_size']
else:
    config['enc_voc_size'] = config['voc_size']
    config['dec_voc_size'] = config['enc_voc_size']

samples  = len(train_set['source'])
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


train_data        = build_data(train_set)
train_data_plain  = zip(*(train_set['source'], train_set['target']))
test_data_plain   = zip(*(test_set['source'],  test_set['target']))

# train_data_plain  = zip(*(train_set['source'], train_set['target']))
# test_data_plain   = zip(*(test_set['source'],  test_set['target']))

train_size        = len(train_data_plain)
test_size         = len(test_data_plain)
tr_idx            = n_rng.permutation(train_size)[:2000].tolist()
ts_idx            = n_rng.permutation(test_size )[:2000].tolist()
logger.info('load the data ok.')
notrain           = False

# build the agent
if config['copynet']:
    agent  = NRM(config, n_rng, rng, mode=config['mode'],
                 use_attention=True, copynet=config['copynet'], identity=config['identity'])
else:
    agent  = NRM0(config, n_rng, rng, mode=config['mode'],
                  use_attention=True, copynet=config['copynet'], identity=config['identity'])

agent.build_()
if notrain:
    agent.compile_('display')
else:
    agent.compile_('all')
print 'compile ok.'

# load the model
# agent.load(config['path_h5'] +
# '/experiments.Copy{2}.id={0}.epoch={1}.pkl'.format('20160229-105153', 1, config['modelname']))

echo   = 0
epochs = 10
skip   = -1  # 25000
if echo > 0:
    tmark = '20160229-105153'  # '20160227-013418'    # copynet multi-source model
    agent.load(config['path_h5'] + '/experiments.Copy{2}.id={0}.epoch={1}.pkl'.format(tmark, echo, config['modelname']))

while echo < epochs:
    echo += 1
    loss  = []

    def output_stream(dataset, batch_size, size=1):
        data_stream = dataset.get_example_stream()
        data_stream = transformers.Batch(data_stream,
                                         iteration_scheme=schemes.ConstantScheme(batch_size))

        # add padding and masks to the dataset
        data_stream = transformers.Padding(data_stream, mask_sources=('source', 'target'))
        return data_stream

    def prepare_batch(batch, mask, fix_len=None):
        data = batch[mask].astype('int32')
        data = np.concatenate([data, np.zeros((data.shape[0], 1), dtype='int32')], axis=1)

        def cut_zeros(data, fix_len=None):
            if fix_len is not None:
                return data[:, : fix_len]
            for k in range(data.shape[1] - 1, 0, -1):
                data_col = data[:, k].sum()
                if data_col > 0:
                    return data[:, : k + 2]
            return data
        data = cut_zeros(data, fix_len)
        return data

    def cc_martix(source, target):
        cc = np.zeros((source.shape[0], target.shape[1], source.shape[1]), dtype='float32')
        for k in xrange(source.shape[0]):
            for j in xrange(target.shape[1]):
                for i in xrange(source.shape[1]):
                    if (source[k, i] == target[k, j]) and (source[k, i] > 0):
                        cc[k][j][i] = 1.
        return cc

    def unk_filter(data):
        if config['voc_size'] == -1:
            return copy.copy(data)
        else:
            mask = (np.less(data, config['voc_size'])).astype(dtype='int32')
            data = copy.copy(data * mask + (1 - mask))
            return data

    # training
    if not notrain:
        train_batches = output_stream(train_data, config['batch_size']).get_epoch_iterator(as_dict=True)
        logger.info('\nEpoch = {} -> Training Set Learning...'.format(echo))
        progbar = Progbar(train_size / config['batch_size'])
        for it, batch in enumerate(train_batches):

            # skip some iterations
            if echo == 1  and it < skip:
                continue

            # obtain data
            data_s = prepare_batch(batch, 'source')
            data_t = prepare_batch(batch, 'target')
            if config['copynet']:
                data_c = cc_martix(data_s, data_t)
                # data_c = prepare_batch(batch, 'target_c', data_t.shape[1])
                loss += [agent.train_(unk_filter(data_s), unk_filter(data_t), data_c)]
            else:
                loss += [agent.train_(unk_filter(data_s), unk_filter(data_t))]

            progbar.update(it, [('loss_reg', loss[-1][0]), ('ppl.', loss[-1][1])])

            if it % 200 == 0:
                logger.info('Echo={} Evaluation Sampling.'.format(it))
                logger.info('generating [training set] samples')
                for _ in xrange(5):
                    idx              = int(np.floor(n_rng.rand() * train_size))
                    train_s, train_t = train_data_plain[idx]
                    v                = agent.evaluate_(np.asarray(train_s, dtype='int32'),
                                                       np.asarray(train_t, dtype='int32'),
                                                       idx2word,
                                                       np.asarray(unk_filter(train_s), dtype='int32'))
                    print '*' * 50

                logger.info('generating [testing set] samples')
                for _ in xrange(5):
                    idx            = int(np.floor(n_rng.rand() * test_size))
                    test_s, test_t = test_data_plain[idx]
                    v              = agent.evaluate_(np.asarray(test_s, dtype='int32'),
                                                     np.asarray(test_t, dtype='int32'),
                                                     idx2word,
                                                     np.asarray(unk_filter(test_s), dtype='int32'))
                    print '*' * 50

            # save the weights.
            if it % 5000 == 0:
                agent.save(config['path_h5'] + '/experiments.Copy{2}.id={0}.epoch={1}.pkl'.format(tmark, echo, config['modelname']))

            if (it % 5000 == 0) and it > 0:
                print 'testing accuracy !!'

                def analysis_(data_plain, t_idx, mode='Training'):
                    progbar_tr = Progbar(2000)
                    print '\n' + '__' * 50
                    cpy, cpy_pos = 0, 0
                    for it, idx in enumerate(t_idx):
                        train_s, train_t = data_plain[idx]
                        c = float(agent.analyse_(np.asarray(train_s, dtype='int32'),
                                           np.asarray(train_t, dtype='int32'),
                                           idx2word))
                        # copy mode
                        cpy     += 1
                        cpy_pos += c
                        progbar_tr.update(it + 1, [('Copy', cpy_pos)])
                    logger.info('\n{0} Accuracy:' +
                                '\t{1}/{2} = {3}%'.format(mode, cpy_pos, cpy, 100 * cpy_pos / float(cpy)))
                    print '==' * 50

                # analysis_(train_data_plain, tr_idx, 'Training')
                analysis_(test_data_plain,  ts_idx, 'Testing')



