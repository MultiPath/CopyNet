__author__ = 'jiataogu'

import logging

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from emolga.models.encdec import RNNLM, AutoEncoder
from emolga.models.variational import Helmholtz, VAE, HarX, THarX, NVTM
# from models.ntm_encdec import RNNLM, AutoEncoder, Helmholtz, BinaryHelmholtz
from emolga.utils.generic_utils import *
from emolga.dataset.build_dataset import deserialize_from_file, build_fuel, obtain_stream
from emolga.config import setup_ptbz, setup_ptb2
from emolga.config_variant import *

setup = setup_bienc


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
tmark  = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
config = setup()   # load settings.
for w in config:
    print '{0}={1}'.format(w, config[w])

logger = init_logging(config['path_log'] + '/emolga.RHM.id={}.log'.format(tmark))
n_rng  = np.random.RandomState(config['seed'])
np.random.seed(config['seed'])
rng    = RandomStreams(n_rng.randint(2 ** 30))

logger.info('Start!')

# load the dataset and build a fuel-dataset.
idx2word, word2idx = deserialize_from_file(config['vocabulary_set'])
config['enc_voc_size'] = max(zip(*word2idx.items())[1]) + 1
config['dec_voc_size'] = config['enc_voc_size']
logger.info('build dataset done. vocabulary size = {0}/ batch size = {1}'.format(
        config['dec_voc_size'], config['batch_size']))

# training & valid & tesing set.
train_set, train_size = build_fuel(deserialize_from_file(config['dataset']))
valid_set, valid_size = build_fuel(deserialize_from_file(config['dataset_test']))  # use test set for a try

# weiget save.
savefile = config['path_h5'] + '/emolga.RHM.id={}.h5'.format(tmark)

# build the agent
if config['model'] == 'RNNLM':
    agent = RNNLM(config, n_rng, rng, mode=config['mode'])
elif config['model'] == 'HarX':
    agent = THarX(config, n_rng, rng, mode=config['mode'])
elif config['model'] == 'Helmholtz':
    agent = Helmholtz(config, n_rng, rng, mode=config['mode'])
else:
    raise NotImplementedError

agent.build_()
agent.compile_('train')
print 'compile ok'

# learning to speak language.
count  = 1000
echo   = 0
epochs = 50
while echo < epochs:
    echo   += 1
    loss    = []
    correct = 0
    scans   = 0

    # visualization the embedding weights.
    # if echo > 1:
    #    plt.figure(3)
    #    visualize_(plt.subplot(111), agent.decoder.Embed.get_params()[0].get_value(), name='encoder embedding',
    #  text=idx2word)
    #    plt.show()

    # if not config['use_noise']:

    # training
    train_batches = obtain_stream(train_set, config['batch_size']).get_epoch_iterator(as_dict=True)
    valid_batches = obtain_stream(valid_set, config['eval_batch_size']).get_epoch_iterator(as_dict=True)

    def prepare_batch(batch):
        data = batch['data'].astype('int32')
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
    logger.info('Epoch = {} -> Training Set Learning...'.format(echo))
    progbar = Progbar(train_size / config['batch_size'])
    for it, batch in enumerate(train_batches):
        # get data
        data = prepare_batch(batch)
        if config['model'] == 'RNNLM' or config['model'] == 'AutoEncoder':
            loss.append(agent.train_(data, config['repeats']))
            progbar.update(it, [('loss_reg', loss[-1][0]), ('ppl.', loss[-1][1])])
        elif config['model'] == 'Helmholtz' or 'HarX':
            loss.append(agent.train_(data, config['repeats']))
            weightss = np.sum([np.sum(abs(w)) for w in agent.get_weights()])
            progbar.update(it, [('lossPa', loss[-1][0]), ('lossPxa', loss[-1][1]), ('lossQ', loss[-1][2]),
                                ('perplexity', np.log(loss[-1][3])), ('NLL', loss[-1][4]), ('L1', weightss)])

        """
        watch = agent.watch(data)
        print '.'
        pprint(watch[0][0])
        pprint(watch[2][0])
        # pprint(watch[2][0])
        sys.exit(111)
        """

        # if it % 100 == 50:
        #     sys.exit(-1)
        #     # print '.'
        #     # print 'encoded = {}'.format(encoded[11])
        #     # print 'mean = {}'.format(mean[11])
        #     # print 'std = {}'.format(std[11])
        #
        #     # watch = agent.watch(data)
        #     # print '.'
        #     # print 'train memory {}'.format(watch[0][0])
        #
        #     for kk in xrange(5):
        #         # sample a sentence.
        #         # action        = agent.action_sampler()
        #         # context       = agent.context_trans(action)
        #         if config['model'] is 'AutoEncoder':
        #             source  = data[kk][None, :]
        #             truth   = ' '.join(print_sample(idx2word, source[0].tolist())[:-1])
        #             print '\ntruth: {}'.format(truth)
        #             context = agent.memorize(source)
        #             sample, score = agent.generate_(context, max_len=data.shape[1])
        #         else:
        #             sample, score = agent.generate_(max_len=data.shape[1])
        #
        #         if sample[-1] is not 0:
        #             sample += [0]  # fix the end.
        #         question = ' '.join(print_sample(idx2word, sample)[:-1])
        #         print '\nsample: {}'.format(question)
        #         print 'PPL: {}'.format(score)
        #         scans   += 1.0

    print ' </s>.'
    logger.info('Epoch = {0} finished.'.format(echo))

    # validation
    logger.info('Epoch = {} -> Vadlidation Set Evaluation...'.format(echo))
    progbar = Progbar(valid_size / config['batch_size'])
    for it, batch in enumerate(valid_batches):
        # get data
        data = prepare_batch(batch)
        if config['model'] == 'Helmholtz' or 'HarX':
            loss.append(agent.evaluate_(data))
            progbar.update(it, [('NLL', loss[-1][0]), ('perplexity', np.log(loss[-1][1]))])
        else:
            raise NotImplementedError

    print ' </s>.'
    # save the weights.
    agent.save(config['path_h5'] + '/emolga.RHM.id={0}.epoch={1}.pkl'.format(tmark, echo))

    # logger.info('Learning percentage: {}'.format(correct / scans))


# inference test
# batches = data_stream.get_epoch_iterator(as_dict=True)
# for it, batch in enumerate(batches):
#     data = batch['data'].astype('int32')
#     data = np.concatenate([data, np.zeros((data.shape[0], 1), dtype='int32')], axis=1)
#     mean, std = agent.inference_(data)
#     print mean
#     break
# print count