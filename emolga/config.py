__author__ = 'jiataogu'
import os
import os.path as path

def setup_ptb2():
    # pretraining setting up.
    # get the lm_config.

    config = dict()
    config['on_unused_input'] = 'ignore'
    config['seed']            = 3030029828
    config['level']           = 'DEBUG'

    # config['model']           = 'RNNLM'
    # config['model']           = 'VAE'
    # config['model']           = 'RNNLM' #'Helmholtz'
    config['model']           = 'HarX'
    config['highway']         = False
    config['use_noise']       = False

    config['optimizer']       = 'adam'  #'adadelta'
    # config['lr']              = 0.1

    # config['optimizer']       = 'sgd'

    # dataset
    config['path']            = path.realpath(path.curdir) + '/'  # '/home/thoma/Work/Dial-DRL/'
    config['vocabulary_set']  = config['path'] + 'dataset/ptbcorpus/voc.pkl'
    config['dataset']         = config['path'] + 'dataset/ptbcorpus/data_train.pkl'
    config['dataset_valid']   = config['path'] + 'dataset/ptbcorpus/data_valid.pkl'
    config['dataset_test']    = config['path'] + 'dataset/ptbcorpus/data_test.pkl'
    # output hdf5 file place.
    config['path_h5']         = config['path'] + 'H5'
    if not os.path.exists(config['path_h5']):
        os.mkdir(config['path_h5'])

    # output log place
    config['path_log']        = config['path'] + 'Logs'
    if not os.path.exists(config['path_log']):
        os.mkdir(config['path_log'])

    # size
    config['batch_size']      = 20
    config['eval_batch_size'] = 20
    config['mode']            = 'RNN'  # NTM
    config['binary']          = False

    # Encoder: dimension
    config['enc_embedd_dim']  = 300
    config['enc_hidden_dim']  = 300
    config['enc_contxt_dim']  = 350
    config['encoder']         = 'RNN'
    config['pooling']         = False

    # Encoder: Model
    config['bidirectional']   = False  # True
    config['decposterior']    = True
    config['enc_use_contxt']  = False

    # Agent: dimension
    config['action_dim']      = 50
    config['output_dim']      = 300

    # Decoder: dimension
    config['dec_embedd_dim']  = 300
    config['dec_hidden_dim']  = 300
    config['dec_contxt_dim']  = 300

    # Decoder: Model
    config['shared_embed']    = False
    config['use_input']       = False
    config['bias_code']       = False   # True
    config['dec_use_contxt']  = True
    config['deep_out']        = False
    config['deep_out_activ']  = 'tanh'  # maxout2
    config['bigram_predict']  = False
    config['context_predict'] = True    # False
    config['leaky_predict']   = False   # True
    config['dropout']         = 0.3

    # Decoder: sampling
    config['max_len']         = 88  # 15
    config['sample_beam']     = 10
    config['sample_stoch']    = False
    config['sample_argmax']   = False

    # Auto-Encoder
    config['nonlinear_A']     = True
    config['nonlinear_B']     = False

    # VAE/Helmholtz: Model
    config['repeats']         = 10
    config['eval_repeats']    = 10
    config['eval_N']          = 10

    config['variant_control'] = False
    config['factor']          = 10.
    config['mult_q']          = 10.

    print 'setup ok.'
    return config


