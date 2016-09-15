# coding=utf-8
__author__ = 'jiataogu'

import logging

import theano
from matplotlib import pyplot
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from config import setup, setup_main
from dataset import deserialize_from_file, divide_dataset, build_fuel, GuessOrder
from game.asker import Asker
from game.responder import Responder
from models.variational import Helmholtz
from utils.generic_utils import *

theano.config.optimizer = 'fast_compile'
Asker       = Asker  # GridAsker # PyramidAsker  # GridAsker
logger      = logging.getLogger(__name__)
lm_config   = setup()
main_config = setup_main() # setup_grid6()  # setup_pyramid()  # setup_grid()  # setup_main()
# logging.basicConfig(level= main_config['level'], format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

np.random.seed(main_config['seed'])
n_rng  = np.random.RandomState(main_config['seed'])
rng    = RandomStreams(n_rng.randint(2 ** 30), use_cuda=True)

"""
Main Loop.
"""
print 'start.'

# load the dataset and build a fuel-dataset.
idx2word, word2idx = deserialize_from_file(lm_config['vocabulary_set'])

# load the fake_dialogue dataset.
print 'Dataset: {}'.format(main_config['fake_diag'])
fake_data = deserialize_from_file(main_config['fake_diag'])
train_set, test_set = divide_dataset(fake_data, main_config['test_size'], 200000)

lm_config['enc_voc_size']   = max(zip(*word2idx.items())[1]) + 1
lm_config['dec_voc_size']   = lm_config['enc_voc_size']
lm_config['state_dim']      = main_config['core_hidden_dim']
main_config['enc_voc_size'] = lm_config['enc_voc_size']

database           = deserialize_from_file(lm_config['dataset'])
dataset            = build_fuel(database)
weights_file       = lm_config['weights_file']
answer_templates   = {0: 'I cannot understand.', 1: 'Congrats!', 2: 'Pity.'}

logger.info('build dataset done. vocabulary size = {0}'.format(lm_config['dec_voc_size']))

start_time         = time.time()
# build the environment
game               = GuessOrder(rng=n_rng, size=main_config['game_length'])
environment        = Responder(game=game)

# load the pretrained generator
generator          = Helmholtz(lm_config, n_rng, rng, dynamic_prior=True)
generator.build_()
generator.load(weights_file)
generator.dynamic()

# build the agent.
agent              = Asker(main_config, lm_config, n_rng, rng, generator)
agent.build_()
agent.compile_asker()
logger.info('compile the asker sampler ok.')

# # build the scheduled trainer if any.
# agent.compile_scheduled_trainer()
# logger.info('compile the asker ss-learner ok.')

# build the trainer
agent.compile_trainer()
logger.info('compile the asker learner ok.')

end_time           = time.time()
logger.info('compiling done. It costs {} seconds'.format(end_time - start_time))


def simulator(M=25, display=False):
    """
    Dialogue Simulation
    """
    start_time = time.time()
    progbar    = Progbar(M)
    logger.info('Start simulation.')
    train_data = {'X': [], 'Y': [], 'A': [], 'R': [], 'G': [], 'T': [], 'text': [], 'acc': []}
    for ep in xrange(M):
        environment.reset()
        episode            = {'x': [], 'y': [], 'a': [], 'r': []}

        conversation       = ''
        conversation      += '\n\n\n' + '***' * 30
        conversation      += '\nGame start.'

        turn               = 0
        maxturn            = 16
        kwargs             = {'turn': turn, 'maxturn': maxturn}
        for k in xrange(maxturn + 1):
            if kwargs['turn'] == maxturn:
                guess, score   = agent.act(kwargs)
                conversation  += '\n' + '_' * 93 + '[{}]'.format(kwargs['turn'])
                conversation  += '\n(´✪ ‿ ✪`)ノ : {}'.format('My answer = ' + ' '.join([str(w) for w in guess]))

                corrects       = environment.get_answer()
                conversation  += '\n{:>78} : ლ（´∀`ლ）'.format(' '.join([str(w) for w in corrects]))

                Accuracy       = sum([g == c for g, c in zip(guess, corrects)]) / float(len(guess))
                conversation  += '\n{:>78} : ლ（´∀`ლ）'.format('Accuracy = {}%'.format(Accuracy * 100))

                episode['g'] = np.asarray(guess)
                episode['t'] = np.asarray(corrects)
                episode['r'].append(Accuracy)
                episode['c'] = Accuracy
                break

            next_action, next_sent, kwargs  = agent.act(kwargs)
            question           = ' '.join(print_sample(idx2word, next_sent)[:-1])
            conversation      += '\n' + '_' * 93 + '[{}]'.format(kwargs['turn'])
            conversation      += '\n(´◉ ω ◉`)？ : {}'.format(question)

            got                = environment.parse(question)
            reward             = 0 if got > 0 else -1
            kwargs['prev_asw'] = np.asarray([got], dtype='int32')
            conversation += '\n{:>78} : (●´ε｀●)'.format(answer_templates[got])

            # registration
            episode['a'].append(next_action)
            episode['y'].append(next_sent[None, :])
            episode['x'].append(got)
            episode['r'].append(reward)

        conversation += '\nGame End\n' + '***' * 30

        if display:
            logger.info(conversation)

        # concatenate
        train_data['A'].append(np.concatenate(episode['a'], axis=0)[None, :, :])
        train_data['Y'].append(np.concatenate(episode['y'], axis=0)[None, :, :])
        train_data['X'].append(np.asarray(episode['x'], dtype='int32')[None, :])
        train_data['R'].append(np.asarray(episode['r'], dtype='float32')[::-1].cumsum()[::-1][None, :])
        train_data['G'].append(episode['g'][None, :])
        train_data['T'].append(episode['t'][None, :])
        train_data['text'].append(conversation)
        train_data['acc'].append(episode['c'])

        progbar.update(ep + 1, [('accuracy', episode['c'])])

    train_data['A'] = np.concatenate(train_data['A'], axis=0).astype('float32')
    train_data['X'] = np.concatenate(train_data['X'], axis=0).astype('int32')
    train_data['Y'] = np.concatenate(train_data['Y'], axis=0).astype('int32')
    train_data['R'] = np.concatenate(train_data['R'], axis=0).astype('float32')
    train_data['G'] = np.concatenate(train_data['G'], axis=0).astype('int32')
    train_data['T'] = np.concatenate(train_data['T'], axis=0).astype('int32')

    end_time = time.time()
    print ''
    logger.info('Simulation {0} eposides with {1} seconds.'.format(M, end_time - start_time))
    return train_data


def learner(data, fr=1., fs=1., fb=1.):
    """
    Training.
    """
    start_time = time.time()
    X     = data['X']   # answers obtained from the environment;
    Y     = data['Y']   # questions generated based on policy;
    A     = data['A']   # actions performed in Helmholtz questions generator;
    R     = data['R']   # cumulative reward obtained through conversation;
    guess = data['G']   # final guess order given by the agent
    truth = data['T']   # real order in the environment

    loss  = agent.train(X, Y, A, R, guess, truth, fr, fs, fb)
    end_time = time.time()
    logger.info('Training this batch with {0} seconds.'.format(end_time - start_time))
    logger.info('REINFORCE Loss = {0}, Supervised loss = {1}, Baseline loss = {2}'.format(
        loss[0], loss[1], loss[2]))
    return loss


def SL_learner(data, batch_size=25, eval_freq=0, eval_train=None, eval_test=None):
    """
    Supervised Learning with fake-optimal logs.
    One epoch for all data.
    """
    start_time = time.time()
    X          = data['X'].astype('int32')   # answers obtained from the environment;
    Y          = data['Y'].astype('int32')   # questions generated based on policy;
    T          = data['T'].astype('int32')   # real order in the environment

    # index shuffle
    idx        = np.arange(X.shape[0]).tolist()
    np.random.shuffle(idx)

    num_batch  = X.shape[0] / batch_size
    progbar    = Progbar(num_batch)
    batch_from = 0
    loss       = []

    if eval_freq > 0:
        eval_batch  = num_batch / eval_freq
        eval_start  = 0

        batches     = []

        accs, unds  = [], []
        acct, undt  = [], []

    for batch in xrange(num_batch):
        batch_to    = batch_from + batch_size
        if batch_to > X.shape[0]:
            batch_to = X.shape[0]

        batch_X     = X[idx[batch_from: batch_to]]
        batch_Y     = Y[idx[batch_from: batch_to]]
        batch_T     = T[idx[batch_from: batch_to]]

        if not main_config['multi_task']:
            loss.append(agent.train_sl(batch_X, batch_Y, batch_T))
            # if not main_config['ssl']:
            #     loss.append(agent.train_sl(batch_X, batch_Y, batch_T))
            # else:
            #     loss.append(agent.train_ssl(batch_X, batch_Y, batch_T, 3, 10.))
            progbar.update(batch + 1, [('loss', loss[-1])])
        else:
            loss.append(agent.train_sl(batch_X, batch_Y, batch_T))
            # loss.append(agent.train_mul(batch_X, batch_Y, batch_T, 3, 10.))
            progbar.update(batch + 1, [('loss', loss[-1][0]), ('ppl', loss[-1][1]), ('asw loss', loss[-1][2])])
        batch_from  = batch_to

        if eval_freq > 0:
            eval_start += 1
            if eval_start == eval_batch or batch == num_batch - 1:
                batches.append(batch_to)
                if eval_train:
                    logger.info('\ntesting on sampled training set.')
                    acc, und = SL_test(eval_train)
                    accs.append(acc)
                    unds.append(und)

                if eval_train:
                    logger.info('testing on sampled testing set.')
                    acc, und = SL_test(eval_test)
                    acct.append(acc)
                    undt.append(und)
                eval_start   = 0

    end_time   = time.time()
    logger.info('Training this epoch with {0} seconds.'.format(end_time - start_time))
    logger.info('Supervised loss = {}'.format(np.mean(loss)))
    if eval_freq > 0:
        eval_details = {'batch_id': batches, 'acc_train': accs, 'acc_test': acct, 'und_train': unds, 'und_test': undt}
        return loss, eval_details
    return loss


def main():
    losses   = []
    accuracy = []
    for echo in xrange(4000):
        logger.info('Iteration = {}'.format(echo))
        train_data = simulator(M=20)

        print train_data['text'][-1]

        loss       = learner(train_data, fr=0.)
        losses.append(loss)
        accuracy  += train_data['acc']

        if echo % 100 == 99:
            plt.plot(accuracy)
            plt.show()

    # pkl.dump(losses, open('losses.temp.pkl'))


def check_answer(x, y, g):
    g     = np.asarray(g)
    environment.game.set_answer(g)
    s     = 0
    for k in xrange(x.shape[1]):
        question           = ' '.join(print_sample(idx2word, y[0][k].tolist())[:-1])
        got                = environment.parse(question)
        if got == 2 - x[0][k]:
            s += 1.
    return s / x.shape[1]


def display_session(x, y, g, t, acc, cov):
    """
    display a dialogue session
    """
    conversation       = ''
    conversation      += '\n\n\n' + '***' * 30
    conversation      += '\nGame start.'

    for k in xrange(x.shape[1]):
        question           = ' '.join(print_sample(idx2word, y[0][k].tolist())[:-1])
        conversation      += '\n' + '_' * 93 + '[{}]'.format(k + 1)
        conversation      += '\n(´◉ ω ◉`)？ : {}'.format(question)
        got                = x[0][k]
        conversation += '\n{:>78} : (●´ε｀●)'.format(answer_templates[got])

    conversation  += '\n' + '_' * 93 + '[{}]'.format(k + 1)
    conversation  += '\n(´✪ ‿ ✪`)ノ : {}'.format('My answer = ' + ' '.join([str(w) for w in g]))
    conversation  += '\n{:>78} : ლ（´∀`ლ）'.format(' '.join([str(w) for w in t[0]]))
    conversation  += '\n{:>78} : ლ（´∀`ლ）'.format('Accuracy = {}%'.format(acc * 100))
    conversation  += '\n{:>78} : ლ（´∀`ლ）'.format('Understand = {}%'.format(cov * 100))
    conversation  += '\nGame End\n' + '***' * 30
    return conversation


def SL_test(test_set):
    print '...'
    progbar    = Progbar(main_config['test_size'])
    accuracy   = []
    understand = []
    # untruth    = []
    at         = 0
    for k in xrange(main_config['test_size']):
        at        += 1
        x          = test_set['X'][None, k]
        y          = test_set['Y'][None, k]
        t          = test_set['T'][None, k]

        g, _, acc  = agent.evaluate(x, y, t)
        cov        = check_answer(x, y, g)
        # cov_t      = check_answer(x, y, t[0].tolist())
        progbar.update(at, [('acc', acc), ('und', cov)])
        # untruth.append(cov_t)
        accuracy.append(acc)
        understand.append(cov)

    acc        = np.mean(accuracy)
    und        = np.mean(understand)
    print '\nevaluation.. avarage accuracy = {0}% /understand {1}% questions'.format(
        100 * acc, 100 * und)
    # print 'check truth {}%'.format(100 * np.mean(untruth))
    return acc, und


def main_sl():
    # get the evaluation set.
    evaluation_set  = n_rng.randint(0, train_set['X'].shape[0], main_config['test_size']).tolist()
    eval_train      = dict()
    eval_train['X'] = train_set['X'][evaluation_set]
    eval_train['Y'] = train_set['Y'][evaluation_set]
    eval_train['T'] = train_set['T'][evaluation_set]

    eval_test       = test_set
    eval_details    = {'batch_id': [], 'acc_train': [],
                       'acc_test': [], 'und_train': [], 'und_test': []}

    for echo in xrange(500):
        logger.info('Epoch = {}'.format(echo))
        loss, ed    = SL_learner(train_set, batch_size=50, eval_freq=10,
                                 eval_train=eval_train, eval_test=eval_test)
        eval_details['acc_train'] += ed['acc_train']
        eval_details['acc_test']  += ed['acc_test']
        eval_details['und_train'] += ed['und_train']
        eval_details['und_test']  += ed['und_test']
        eval_details['batch_id']  += [(t + 200000 * echo) / 1000.0 for t in ed['batch_id']]

        pyplot.figure(1)
        pyplot.plot(eval_details['batch_id'], eval_details['acc_train'], 'b')
        pyplot.plot(eval_details['batch_id'], eval_details['acc_test'], 'r')
        pyplot.xlabel('iterations (x 1000)')
        pyplot.ylabel('accuracy')
        pyplot.savefig('./acc-gru.png')

        pyplot.figure(2)
        pyplot.plot(eval_details['batch_id'], eval_details['und_train'], 'b')
        pyplot.plot(eval_details['batch_id'], eval_details['und_test'], 'r')
        pyplot.xlabel('iterations (x 1000)')
        pyplot.ylabel('understand rate')
        pyplot.savefig('./und-gru.png')

        logger.info("saving ok!")
        # if echo % 20 == 19:
        #     pyplot.figure(1)
        #     pyplot.plot(acc_s, 'r')
        #     pyplot.plot(acc_t, 'g')
        #     pyplot.figure(2)
        #     pyplot.plot(und_s, 'r')
        #     pyplot.plot(und_t, 'g')
        #     pyplot.show()

# agent.main_config['sample_beam']   = 1
# agent.main_config['sample_argmax'] = True
main_sl()
