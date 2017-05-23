#coding=utf8
'''
    the module that schedule the experiments for different datasets and configs
'''
import time
import logging
import sys
import argparse

import numpy as np
import yaml

from fm_anova_kernel_glasso import FMAKGL

from data_util import DataLoader
from logging_util import init_logger

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sn', help='split number, which specify the split data to run, default is 1', default=1, type=int)
    parser.add_argument('-K', help='number of latent features when factorizing P, or Q in FM', type=int)
    parser.add_argument('-reg', help='regularization for all parameters, if given, set all reg otherwise doing nothing', type=float)
    parser.add_argument('-reg_P', help='regularization for P', type=float)
    parser.add_argument('-reg_Q', help='regularization for Q', type=float)
    parser.add_argument('-reg_W', help='regularization for W', type=float)
    parser.add_argument('-max_iters', help='max iterations of the training process', type=int)
    parser.add_argument('-eps', help='stopping criterion', type=float)
    parser.add_argument('-eta', help='learning rate in the beginning', type=float)
    parser.add_argument('-bias_eta', help='learning rate for bias', type=float)
    parser.add_argument('-initial', help='initialization of random starting', type=float)
    parser.add_argument('config',  help='specify the config file')
    return parser.parse_args()

def update_configs_by_args(config, args):
    args_dict = vars(args)
    #if reg is specified, set all regularization values to reg
    if args.reg is not None:
        config['reg_W'] = config['reg_P'] = config['reg_Q'] = args.reg
        del args_dict['reg_W']
        del args_dict['reg_P']
        del args_dict['reg_Q']

    for k, v in args_dict.items():
        if v is not None:
            config[k] = v

def update_configs(config, args):
    '''
        1, generate some configs dynamically, according to given parameters
            L, N, exp_id, logger
        2, fix one bug: make 1e-6 to float
        3, create exp data dir, replacing 'dt' with the specified dt
        3, update by arguments parser
    '''
    exp_id = int(time.time())
    config['exp_id'] = exp_id


    L = len(config.get('meta_graphs'))
    config['L'] = L

    F = config['F']
    config['N'] = 2 * L * F
    config['eps'] = float(config['eps'])
    config['initial'] = float(config['initial'])
    config['eta'] = float(config['eta'])
    config['bias_eta'] = float(config['bias_eta'])

    dt = config['dt']
    config['data_dir'] = 'data/%s/exp_split/%s/' % (dt, config['sn'])
    config['train_filename'] = 'ratings_train_%s.txt' % config['sn']
    config['test_filename'] = 'ratings_test_%s.txt' % config['sn']

    update_configs_by_args(config, args)

def set_logfile(config, args):
    log_filename = 'log/fmg_%s_%s_split%s.log' % (config['dt'], config['exp_type'], config['sn'])
    config['log_filename'] = log_filename
    init_logger('', config['log_filename'], logging.INFO, False)

def init_exp_configs(config_filename):
    '''
        load the configs
    '''
    config = yaml.load(open(config_filename, 'r'))
    config['config_filename'] = config_filename
    return config

def run_glasso(config, data_loader):
    print 'run fm glasso..., check the log in %s ...' % config.get('log_filename')
    logging.info('******\n%s\n******', config)
    run_start = time.time()
    fm_ak_gl = FMAKGL(config, data_loader)
    fm_ak_gl.train()
    rmses, maes = fm_ak_gl.get_eval_res()
    cost = (time.time() - run_start) / 3600.0
    logging.info('******config*********\n%s\n******', config)
    logging.info('**********fm_anova_kernel_glasso finish, run once, cost %.2f hours*******\n, rmses: %s, maes: %s\navg rmse=%s, avg mae=%s\n***************', cost, rmses[-5:], maes[-5:], np.mean(rmses[-5:]), np.mean(maes[-5:]))

def run_reg_varying(config, data_loader):

    for reg in [1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0, 100.0]:
        config['reg_W'] = config['reg_P'] = config['reg_Q'] = reg
        run_glasso(config, data_loader)

def run():
    '''
        given the train/test files, run for once to see the results
    '''
    args = get_args()

    config = init_exp_configs(args.config)
    update_configs(config, args)
    set_logfile(config, args)

    data_loader = DataLoader(config)

    if config['exp_type'] == 'vary_reg':
        run_reg_varying(config, data_loader)

def run_regsvd(config, data_loader):
    print 'run RegSVD..., check the log in %s ...' % config.get('log_filename')
    run_start = time.time()
    fm_ak_gl = MF(config, data_loader)
    fm_ak_gl.train()
    rmses1, maes1 = fm_ak_gl.get_eval_res()
    cost1 = (time.time() - run_start) / 3600.0
    logging.info('******config*********\n%s\n******', config)
    logging.info('**********fm_anova_kernel_glasso finish, run once, cost %.2f hours*******\n, rmses: %s, maes: %s\navg rmse=%s, avg mae=%s\n***************', cost1 , rmses1[-5:], maes1[-5:], np.mean(rmses1[-5:]), np.mean(maes1[-5:]))

if __name__ == '__main__':
    run()
