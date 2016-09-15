__author__ = 'jiataogu'
from config import setup_ptb2
setup = setup_ptb2

"""
This file is for small variant fix on original
"""


def setup_bienc(config=None):
    if config is None:
        config = setup()
    print 'make some modification'

    config['bidirectional'] = True
    config['decposterior']  = False
    return config


def setup_dim(config=None):
    if config is None:
        config = setup()
    print 'make some modification'

    config['enc_embedd_dim'] = 300
    config['enc_hidden_dim'] = 300
    config['action_dim']     = 100

    config['dec_embedd_dim'] = 300
    config['dec_hidden_dim'] = 300
    config['dec_contxt_dim'] = 300
    return config


def setup_rep(config=None):
    if config is None:
        config = setup()
    print 'make some modification'

    config['repeats']        = 5
    return config


def setup_opt(config=None):
    if config is None:
        config = setup()
    print 'make some modification'

    config['optimizer']      = 'Adam'
    return config