"""
This is the implementation of Copy-NET
We start from the basic Seq2seq framework for a auto-encoder.
"""
import logging
import time
import numpy as np

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from experiments.config import setup
from emolga.utils.generic_utils import *
from emolga.models.encdec import *
from emolga.dataset.build_dataset import deserialize_from_file
from collections import OrderedDict
from fuel import datasets
from fuel import transformers
from fuel import schemes


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

logger  = init_logging(config['path_log'] + '/experiments.Copy.id={}.log'.format(tmark))
n_rng  = np.random.RandomState(config['seed'])
np.random.seed(config['seed'])
rng    = RandomStreams(n_rng.randint(2 ** 30))
logger.info('Start!')

idx2word, word2idx, idx2word_o, word2idx_o \
        = deserialize_from_file(config['voc'])
idx2word_o[0] = '<eol>'
word2idx_o['<eol>'] = 0

source, target, origin = deserialize_from_file(config['dataset'])
samlpes = len(source)

config['enc_voc_size'] = max(zip(*word2idx.items())[1]) + 1
config['dec_voc_size'] = config['enc_voc_size']
logger.info('build dataset done. ' +
            'dataset size: {} ||'.format(samlpes) +
            'vocabulary size = {0}/ batch size = {1}'.format(
        config['dec_voc_size'], config['batch_size']))


def build_data(source, target):
    # create fuel dataset.
    dataset     = datasets.IndexableDataset(indexables=OrderedDict([('source', source), ('target', target)]))
    dataset.example_iteration_scheme \
                = schemes.ShuffledExampleScheme(dataset.num_examples)
    return dataset, len(source)


train_data, train_size = build_data(source[int(0.2 * samlpes):], target[int(0.2 * samlpes):])
train_data_plain       = zip(*(source[int(0.2 * samlpes):], target[int(0.2 * samlpes):], origin[int(0.2 * samlpes):]))
test_data_plain        = zip(*(source[:int(0.2 * samlpes)], target[:int(0.2 * samlpes)], origin[:int(0.2 * samlpes)]))
test_size              = len(test_data_plain)
logger.info('load the data ok.')

# build the agent
agent  = NRM(config, n_rng, rng, mode=config['mode'],
             use_attention=True, copynet=config['copynet'], identity=config['identity'])
agent.build_()
agent.compile_('all')
print 'compile ok.'

echo   = 0
epochs = 10
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

    def prepare_batch(batch, mask):
        data = batch[mask].astype('int32')
        data = np.concatenate([data, np.zeros((data.shape[0], 1), dtype='int32')], axis=1)

        def cut_zeros(data):
            for k in range(data.shape[1] - 1, 0, -1):
                data_col = data[:, k].sum()
                if data_col > 0:
                    return data[:, : k + 2]
            return data
        data = cut_zeros(data)
        return data

    # training
    train_batches = output_stream(train_data, config['batch_size']).get_epoch_iterator(as_dict=True)
    logger.info('Epoch = {} -> Training Set Learning...'.format(echo))
    progbar = Progbar(train_size / config['batch_size'])
    for it, batch in enumerate(train_batches):
        # obtain data
        data_s, data_t = prepare_batch(batch, 'source'), prepare_batch(batch, 'target')
        loss += [agent.train_(data_s, data_t)]
        progbar.update(it, [('loss_reg', loss[-1][0]), ('ppl.', loss[-1][1])])

        if it % 500 == 0:
            logger.info('generating [training set] samples')
            for _ in xrange(5):
                idx              = int(np.floor(n_rng.rand() * train_size))
                train_s, train_t, train_o = train_data_plain[idx]
                v                = agent.evaluate_(np.asarray(train_s, dtype='int32'),
                                                   np.asarray(train_t, dtype='int32'),
                                                   idx2word, np.asarray(train_o, dtype='int32'), idx2word_o)
                print '*' * 50

            logger.info('generating [testing set] samples')
            for _ in xrange(5):
                idx            = int(np.floor(n_rng.rand() * test_size))
                test_s, test_t, test_o = test_data_plain[idx]
                v              = agent.evaluate_(np.asarray(test_s, dtype='int32'),
                                                 np.asarray(test_t, dtype='int32'),
                                                 idx2word, np.asarray(test_o, dtype='int32'), idx2word_o)
                print '*' * 50
