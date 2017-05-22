#coding=utf8
'''
    Implement the solver factorization machine with group lasso
    version4: gradient v calculation with cython
    version3: gradient calculation with c code for V
    version2: accelerated proximal gradient descent
    version1: proximal gradient descent
'''
import sys
import time
import logging
import ctypes
import threading
from datetime import datetime
import cPickle as pickle
from get_grads_cython import get_grad_v

import numpy as np
from numpy.linalg import norm

from logging_util import init_logger

INCLUDE_RATINGS = True

def init_conifg(dt_arg, reg, exp_type, eps, K=10, F=10):
    global rating_filename
    global logger
    global exp_id
    global dt
    dt = dt_arg

    if dt == 'yelp':
        rating_filename = 'ratings_filter5'
    elif dt in ['yelp-200k', 'yelp-50k', 'yelp-10k', 'yelp-5k', 'yelp-100k']:
        rating_filename = 'ratings'
    elif dt in ['douban']:
        rating_filename = 'ratings'
    elif dt == 'cikm-yelp':
        rating_filename = 'ratings'
    elif dt == 'yelp-sample':
        rating_filename = ''
    elif dt in ['ml-100k', 'ml-1m', 'ml-10m']:
        rating_filename = '%s-rating' % dt
    elif dt == 'amazon-app':
        rating_filename = 'filter5_ratings_Apps_for_Android'
    elif dt in ['amazon-200k', 'amazon-50k', 'amazon-100k', 'amazon-10k', 'amazon-5k']:
        rating_filename = 'ratings'

    if exp_type == 1:
        log_filename = 'log/%s_fm_glasso_once_reg%s_eps%s_K%s_F%s.log' % (dt, reg, eps, K, F)
    elif exp_type == 2:
        log_filename = 'log/%s_fm_glasso_regv%s_eps%s_K%s_F%s.log' % (dt, reg, eps, K, F)

    exp_id = int(time.time())
    logger = init_logger('exp_%s' % exp_id, log_filename, logging.INFO, False)

stf = lambda eta, nw: 1 - eta / nw if eta < nw else 0.0#soft threshold function
stf2 = lambda b: b if b > 0.0 else 0.0#soft threshold function

DEBUG = False

grad_v_lib = ctypes.cdll.LoadLibrary('./cal_grad_v.so')
cal_grad_v = grad_v_lib.cal_grad_v

def prox_op(W, V, eta, gw_inds, gv_inds):
    f_V = V.flatten()
    for i in range(len(gw_inds)):
        W[gw_inds[i]] = stf(eta, norm(W[gw_inds[i]])) * W[gw_inds[i]]
        f_V[gv_inds[i]] = stf(eta, norm(f_V[gv_inds[i]])) * f_V[gv_inds[i]]
    V = f_V.reshape(V.shape)
    return W, V

def group_lasso(P, gp_inds):
    res = 0.0
    for i in range(gp_inds.shape[0]):
        res += norm(P[gp_inds[i]])
    return res

def cal_err(X, Y, W, V, b):
    part1 = np.dot(W, X.T)
    part2 = np.square(np.dot(X, V))
    part3 = np.dot(np.square(X), np.square(V))
    Y_t = b + part1 + 0.5 * (part2 - part3).sum(axis=1)
    return Y_t - Y

def obj(err, W, V, lamb, gw_inds, gv_inds):
    return np.power(err, 2).sum() + lamb * group_lasso(W, gw_inds) + lamb * group_lasso(V.flatten(), gv_inds)

def load_data(t_dir, N, train_filename, test_filename, F):
    start_time = time.time()

    train_data = np.loadtxt(train_filename)
    test_data = np.loadtxt(test_filename)
    train_num = train_data.shape[0]
    test_num = test_data.shape[0]

    uid2reps, bid2reps = load_representation(t_dir, N/2, F)

    X = np.zeros((train_num, N), dtype=np.float64)
    Y = train_data[:,2]
    test_X = np.zeros((test_num, N), dtype=np.float64)
    test_Y = test_data[:,2]

    ind = 0
    for u, b, _ in train_data:
        ur = uid2reps[int(u)]
        br = bid2reps[int(b)]
        X[ind] = np.concatenate((ur,br))
        ind += 1
    X_sparsity = np.count_nonzero(X) * 1.0 / X.size

    ind = 0
    for u, b, _ in test_data:
        ur = uid2reps.get(int(u), np.zeros(N/2, dtype=np.float64))
        br = bid2reps.get(int(b), np.zeros(N/2, dtype=np.float64))
        test_X[ind] = np.concatenate((ur,br))
        ind += 1

    test_X_sparsity = np.count_nonzero(test_X) * 1.0 / test_X.size

    logger.info('finish loading data, cost %.2f seconds, ratings_file=%s, train=%s, test=%s, stat(shape, sparsity): train: (%s, %.4f), test: (%s, %.4f)', time.time() - start_time, rating_filename, train_filename, test_filename, X.shape, X_sparsity, test_X.shape, test_X_sparsity)
    return X, Y, test_X, test_Y

def load_representation(t_dir, fnum, F):
    '''
        load user and item latent features generate by MF for every meta-graph
    '''
    if dt in ['yelp-200k', 'amazon-200k', 'amazon-50k', 'amazon-100k', 'amazon-10k', 'amazon-5k', 'cikm-yelp', 'yelp-50k', 'yelp-10k', 'yelp-5k', 'yelp-100k', 'douban']:
        ufilename = t_dir + 'uids.txt'
        bfilename = t_dir + 'bids.txt'
    uids = [int(l.strip()) for l in open(ufilename, 'r').readlines()]
    uid2reps = {k:np.zeros(fnum, dtype=np.float64) for k in uids}

    bids = [int(l.strip()) for l in open(bfilename, 'r').readlines()]
    bid2reps = {k:np.zeros(fnum, dtype=np.float64) for k in bids}

    if dt == 'yelp-200k':
        ufiles = ['URPSRUB_user.dat', 'URNSRUB_user.dat', 'UPBCatB_top1000_user.dat', 'UPBStarsB_top1000_user.dat', 'UPBStateB_top1000_user.dat', 'UPBCityB_top1000_user.dat', 'UPBUB_top1000_user.dat', 'UNBUB_top1000_user.dat', 'UUB_top1000_user.dat', 'URPARUB_top1000_user.dat', 'URNARUB_top1000_user.dat']
        vfiles = ['URPSRUB_item.dat', 'URNSRUB_item.dat', 'UPBCatB_top1000_item.dat', 'UPBStarsB_top1000_item.dat', 'UPBStateB_top1000_item.dat', 'UPBCityB_top1000_item.dat', 'UPBUB_top1000_item.dat', 'UNBUB_top1000_item.dat', 'UUB_top1000_item.dat', 'URPARUB_top1000_item.dat', 'URNARUB_top1000_item.dat']
    if dt in ['yelp-10k', 'yelp-50k', 'yelp-100k', 'yelp-5k']:
        ufiles = ['URPSRUB_top500_user.dat', 'URNSRUB_top500_user.dat', 'UPBCatB_top500_user.dat', 'UPBStarsB_top500_user.dat', 'UPBStateB_top500_user.dat', 'UPBCityB_top500_user.dat', 'UPBUB_top500_user.dat', 'UNBUB_top500_user.dat', 'UUB_top500_user.dat', 'URPARUB_top500_user.dat', 'URNARUB_top500_user.dat']
        vfiles = ['URPSRUB_top500_item.dat', 'URNSRUB_top500_item.dat', 'UPBCatB_top500_item.dat', 'UPBStarsB_top500_item.dat', 'UPBStateB_top500_item.dat', 'UPBCityB_top500_item.dat', 'UPBUB_top500_item.dat', 'UNBUB_top500_item.dat', 'UUB_top500_item.dat', 'URPARUB_top500_item.dat', 'URNARUB_top500_item.dat']
    elif dt == 'amazon-200k':
        ufiles = ['URPSRUB_user.dat', 'URNSRUB_user.dat', 'UPBCatB_top1000_user.dat', 'UPBBrandB_top1000_user.dat', 'UPBUB_top1000_user.dat', 'UNBUB_top1000_user.dat', 'URPARUB_top1000_user.dat', 'URNARUB_top1000_user.dat']
        vfiles = ['URPSRUB_item.dat', 'URNSRUB_item.dat', 'UPBCatB_top1000_item.dat', 'UPBBrandB_top1000_item.dat', 'UPBUB_top1000_item.dat', 'UNBUB_top1000_item.dat', 'URPARUB_top1000_item.dat', 'URNARUB_top1000_item.dat']
    elif dt in ['amazon-50k','amazon-100k','amazon-10k','amazon-5k']:
        ufiles = ['URPSRUB_top500_user.dat', 'URNSRUB_top500_user.dat', 'UPBCatB_top500_user.dat', 'UPBBrandB_top500_user.dat', 'UPBUB_top500_user.dat', 'UNBUB_top500_user.dat', 'URPARUB_top500_user.dat', 'URNARUB_top500_user.dat']
        vfiles = ['URPSRUB_top500_item.dat', 'URNSRUB_top500_item.dat', 'UPBCatB_top500_item.dat', 'UPBBrandB_top500_item.dat', 'UPBUB_top500_item.dat', 'UNBUB_top500_item.dat', 'URPARUB_top500_item.dat', 'URNARUB_top500_item.dat']
    elif dt == 'cikm-yelp':
        ufiles = ['UPBCatBUB_top500_user.dat', 'UPBCityBUB_top500_user.dat','UNBCatBUB_top500_user.dat', 'UNBCityBUB_top500_user.dat', 'UPBUB_top500_user.dat', 'UNBUB_top500_user.dat', 'UUB_top500_user.dat', 'UCompUB_top500_user.dat']
        vfiles = ['UPBCatBUB_top500_item.dat', 'UPBCityBUB_top500_item.dat','UNBCatBUB_top500_item.dat', 'UNBCityBUB_top500_item.dat', 'UPBUB_top500_item.dat', 'UNBUB_top500_item.dat', 'UUB_top500_item.dat', 'UCompUB_top500_item.dat']
    elif dt == 'douban':
        ufiles = ['UBDBUB_top500_user.dat', 'UBABUB_top500_user.dat', 'UBTBUB_top500_user.dat', 'UGUB_top500_user.dat', 'UBUB_top500_user.dat']
        vfiles = ['UBDBUB_top500_item.dat', 'UBABUB_top500_item.dat', 'UBTBUB_top500_item.dat', 'UGUB_top500_item.dat', 'UBUB_top500_item.dat']

    if INCLUDE_RATINGS:
        ufiles.append('ratings_only_user.dat')
        vfiles.append('ratings_only_item.dat')

    feature_dir = t_dir + 'mf_features/path_count/'
    #exp vary F
    if F != 10:
        feature_dir = t_dir + 'mf_features/path_count/ranks/'
    for find, filename in enumerate(ufiles):
        if F != 10:
            filename = filename.replace('user', 'F%s_user' % F)
            ufiles[find] = filename
        ufs = np.loadtxt(feature_dir + filename, dtype=np.float64)
        cur = find * F
        for uf in ufs:
            uid = int(uf[0])
            f = uf[1:]
            uid2reps[uid][cur:cur+F] = f

    for find, filename in enumerate(vfiles):
        if F != 10:
            filename = filename.replace('item', 'F%s_item' % F)
            vfiles[find] = filename
        bfs = np.loadtxt(feature_dir + filename, dtype=np.float64)
        cur = find * F
        for bf in bfs:
            bid = int(bf[0])
            f = bf[1:]
            bid2reps[bid][cur:cur+F] = f
    logger.info('run for all, F=%s, len(ufiles)=%s, len(vfiles)=%s, ufiles=%s, vfiles=%s', len(ufiles), F, len(vfiles), '|'.join(ufiles), '|'.join(vfiles))

    return uid2reps, bid2reps

def cal_rmse(W, V, b, test_X, test_Y):
    err = cal_err(test_X, test_Y, W, V, b)
    num = test_Y.shape[0]
    rmse = np.sqrt(np.square(err).sum() / num)
    return rmse

def cal_mae(W, V, b, test_X, test_Y):
    err = cal_err(test_X, test_Y, W, V, b)
    num = test_Y.shape[0]
    mae = np.abs(err).sum() / num
    return mae

def run(split_num, t_dir, lamb, K, eps, ite, solver='acc', train_filename='', test_filename='', F=10):
    '''
        K: number of latent features in FM
        lamb: regularization
        ite: max iterations regardless of stopping criteira
        eps: stopping criteria
        eta: learning rate
        F: number of latent features in matrix factorization
        L: number of meta-graph
    '''
    global exp_rmses
    global exp_maes
    global threads_finish
    logger.info('start validation %s, exp_dir=%s, train_filename=%s, test_filename=%s', split_num, t_dir, train_filename, test_filename)
    start_time = time.time()
    if dt in ['yelp-200k', 'yelp-50k', 'yelp-10k', 'yelp-100k', 'yelp-5k']:
        L = 11
    elif dt in ['amazon-200k', 'amazon-50k', 'amazon-100k', 'amazon-10k', 'amazon-5k']:
        L = 8
    elif dt == 'cikm-yelp':
        L = 8
    elif dt == 'douban':
        L = 5
    #F = F
    if INCLUDE_RATINGS:
        L += 1
    N = 2 * L * F
    eta = 1e-7
    eta_b = eta#learning rate
    eta_w = eta#learning rate
    eta_v = eta#learning rate
    beta = 0.9#parameter used in line search
    exp_id = int(time.time())

    b = 0 # bias
    initial = 1e-5
    W = np.random.rand(N).astype(dtype=np.float64) * initial # 1 by N
    V = np.random.rand(N, K).astype(dtype=np.float64) * initial# N by K
    X, Y, test_X, test_Y = load_data(t_dir, N, train_filename, test_filename, F)

    exp_info = 'exp on large scale data, 1-5 scale, when reg=%s, initial=%s, solver=%s' % (lamb, initial, solver)
    exp_time = datetime.now().strftime('%Y-%m-%d %H:%M')
    logger.info('*************exp_time:%s, exp_id=%s, %s*************', exp_time, exp_id, exp_info)
    logger.info('fm_group_lasso started, exp_id=%s, %s group features, solver=%s, config K,F,reg,ite,eta,eps,initial=(%s,%s,%s,%s,(%s,%s,%s),%s,%s)', int(exp_id), L, solver, K, F, lamb, ite, eta_b, eta_w, eta_v, eps, initial)

    gw_inds = np.arange(N).reshape(2*L, F)
    gv_inds = np.arange(N*K).reshape(2*L, F*K)

    if solver == 'acc':
        rmses, maes = train_acc_prox_gradient(t_dir, X, Y, W, V, b, gw_inds, gv_inds, test_X, test_Y, ite, lamb, eta_b, eta_w, eta_v, beta, eps, exp_id, N, K)
    else:
        rmses, maes = train_prox_gradient(t_dir, X, Y, W, V, b, gw_inds, gv_inds, test_X, test_Y, ite, eta, beta, eps, exp_id, N, K)

    total_cost = (time.time() - exp_id) / 3600.0
    logger.info('fm_group_lasso finished, total_cost=%.2f hours exp_time=%s, exp_id=%s, %s group features, solver=%s, config K,F,reg,ite,eta,eps,initial=(%s,%s,%s,%s,(%s,%s,%s),%s,%s)', total_cost, exp_time, int(exp_id), L, solver, K, F, lamb,ite, eta_b, eta_w, eta_v, eps, initial)

    if exp_type > 1:
        round_rmse = np.mean(rmses[-5:])
        round_mae = np.mean(maes[-5:])
        exp_rmses[split_num] = round_rmse
        exp_maes[split_num] = round_mae
        logger.info('finish validation %s, exp_dir=%s, cost %.2f minutes, rmse=%.4f, mae=%.4f', split_num, t_dir, (time.time() - start_time) / 60.0, exp_rmses[split_num], exp_maes[split_num])
        threads_finish[split_num - 1] = True
    return rmses, maes

def cal_grad_v_by_c(err, X_i, part_f, tmp):
    xn = X_i.size
    nc = ctypes.c_int(xn)

    res = np.array([0.0], dtype=np.float64)
    resp = res.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    ep = err.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    xp = X_i.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    fp = part_f.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    tp = tmp.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    cal_grad_v(ep, xp, fp, tp, nc, resp)
    return res[0]

def line_search(err, W, V, b, lamb, eta, gw_inds, gv_inds, obj_v, X, Y, ln, N, K):

    grad_start = time.time()
    grad_w = 2 * np.dot(err, X)
    w_cost = time.time() - grad_start

    part = np.dot(X, V)
    X_square = np.square(X)
    v_start = time.time()
    M = X.shape[0]
    grad_v = get_grad_v(err, W, V, X, part, X_square, N, K, M)
    v_cost = time.time() - v_start

    #print 'grad_w/square/set/grads cost: %.2fs/%.2fs/%.2fs/%.2fs' % (w_cost, square_cost, v_set_cost, time.time() - grad_start)
    #line search with accelerated proximal operator
    for lt in range(ln+1):
        tW, tV = W - eta * grad_w, V - eta * grad_v
        W_p, V_p = prox_op(tW, tV, eta * lamb, gw_inds, gv_inds)
        l_err = cal_err(X, Y, W_p, V_p, b)
        l_obj = obj(l_err, W_p, V_p, lamb, gw_inds, gv_inds)
        if l_obj < obj_v:
            eta = 1.5 * eta
            break
        else:
            eta = 0.9 * eta
    return eta, lt, l_obj, W_p, V_p, v_cost

def train_acc_prox_gradient(t_dir, X, Y, W, V, b, gw_inds, gv_inds, test_X, test_Y, ite, lamb, eta_b, eta_w, eta_v, beta, eps, exp_id, N, K):
    '''
        accelerated proximal gradient method
    '''
    objs = [None] * (ite + 1)
    err = cal_err(X, Y, W, V, b)
    objs[0] = obj(err, W, V, lamb, gw_inds, gv_inds)
    rmses = [cal_rmse(W, V, b, test_X, test_Y)]
    maes = [cal_mae(W, V, b, test_X, test_Y)]

    A = np.hstack((W.reshape(-1,1), V))
    A0, A1, C1 = A.copy(), A.copy(), A.copy()
    c = objs[0]
    r0, r1, q, qeta = 0.0, 1.0, 1.0, 0.5
    eta1 = eta2 = eta_b

    lt1, lt2 = 0, 0
    ln = 100
    for t in range(ite):
        try:
            start = time.time()
            B = A1 + r0/r1 * (C1 - A1) + (r0 - 1)/r1 * (A1 - A0)
            W, V = B[:,0].flatten(), B[:,1:]

            err = cal_err(X, Y, W, V, b)
            obj_b = obj(err, W, V, lamb, gw_inds, gv_inds)

            b = b - eta_b * 2 * err.sum()

            l1start = time.time()
            eta1, lt1, obj_c, W_p, V_p, v_cost1 = line_search(err, W, V, b, lamb, eta1, gw_inds, gv_inds, obj_b, X, Y, ln, N, K)
            l1cost = time.time() - l1start

            if lt1 == ln:
                logger.info('lt1=%s', lt1)
                break

            C1 = np.hstack((W_p.reshape(-1,1), V_p))
            A0 = A1.copy()

            l2cost, v_cost2 = 0.0, 0.0

            if obj_c < c:
                A1 = C1.copy()
                objs[t+1] = obj_c
            else:
                W, V = A1[:,0].flatten(), A1[:,1:]
                err = cal_err(X, Y, W, V, b)
                obj_a = obj(err, W, V, lamb, gw_inds, gv_inds)

                l2start = time.time()
                eta1, lt2, obj_v, W_p, V_p, v_cost2 = line_search(err, W, V, b, lamb, eta1, gw_inds, gv_inds, obj_a, X, Y, ln, N, K)
                l2cost = time.time() - l2start

                if obj_c > obj_v:
                    A1 = np.hstack((W_p.reshape(-1,1), V_p))
                    objs[t+1] = obj_v
                else:
                    A1 = C1.copy()
                    objs[t+1] = obj_c
            if lt2 == ln:
                logger.info('lt2=%s', lt2)
                break

            W, V = A1[:,0].flatten(), A1[:,1:]

            rmses.append(cal_rmse(W, V, b, test_X, test_Y))
            maes.append(cal_mae(W, V, b, test_X, test_Y))
            end = time.time()

            dr = abs(objs[t] - objs[t+1]) / objs[t]
            logger.info('exp_id=%s, iter=%s, (lt1,eta1, v_cost1/cost)=(%s,%s, %.2f/%.2fs), (lt2,eta2,v_cost2/cost)=(%s,%s, %.2f/%.2fs), obj=%.5f(dr=%.8f), rmse=%.5f, mae=%.5f, cost=%.2f seconds', exp_id, t, lt1, eta1, v_cost1, l1cost, lt2, eta2, v_cost2, l2cost, objs[t+1], dr, rmses[t+1], maes[t+1], (end - start))

            r0 = r1
            r1 = (np.sqrt(4 * pow(r0, 2) + 1) + 1) / 2.0
            tq = qeta * q + 1.0
            c = (qeta * q * c + objs[t+1]) / tq
            q = tq

            if  dr < eps:
                break
        except KeyboardInterrupt:
            logger.info('stopped manually, iter=%s,obj=%.5f,rmse=%.5f,mae=%.5f', t, objs[t], rmses[-1], maes[-1])
            break

    split_num = t_dir.split('/')[-2]
    W_wfilename = 'fm_res/split%s_W_%s_exp%s.txt' % (split_num, lamb, exp_id)
    np.savetxt(W_wfilename, W)
    V_wfilename = 'fm_res/split%s_V_%s_exp%s.txt' % (split_num, lamb, exp_id)
    np.savetxt(V_wfilename, V)
    logger.info('W and V saved in %s and %s', W_wfilename, V_wfilename)
    return rmses, maes

def run_5_validation(lamb, K, eps, ite, solver, F=10):
    logger.info('start run_5_validations, dataset=%s, ratings_filename=%s, K=%s,F=%s,eps=%s,reg=%s,iters=%s,solver=%s', dt, rating_filename, K,F,eps,lamb, ite, solver)
    run_start = time.time()
    global exp_rmses
    global exp_maes
    global threads_finish
    exp_maes, exp_rmses = {}, {}
    threads_finish = [False] * 5

    threads = []
    for rnd in xrange(5):
        start_time = time.time()
        t_dir = 'data/%s/exp_split/%s/' % (dt, rnd+1)

        train_filename = t_dir + '%s_train_%s.txt' % (rating_filename, rnd+1)
        test_filename = t_dir + '%s_test_%s.txt' % (rating_filename, rnd+1)

        threads.append(threading.Thread(target=run, args=(rnd+1, t_dir, lamb, K, eps, ite,solver, train_filename, test_filename, F)))

    for t in threads:
        t.daemon = True
        t.start()

    while True:
        time.sleep(1)
        if sum(threads_finish) == 5:
            cost = (time.time() - run_start) / 60.0
            logger.info('**********finish run_5_validations, cost %.2f mins, dataset=%s,rating_filename=%s***********\n*****config: (K, F, reg, eps, iters solver)=(%s, %s, %s, %s, %s, %s), exp rmses: %s, maes: %s\n*******avg rmse=%s, avg mae=%s\n**************', cost, dt, rating_filename, K, F, lamb, eps, ite, solver, exp_rmses.items(), exp_maes.items(), np.mean(exp_rmses.values()), np.mean(exp_maes.values()))
            break

if __name__ == '__main__':
    if len(sys.argv) == 8:
        global exp_type

        dt = sys.argv[1]
        #lambw = float(sys.argv[3].replace('regw',''))
        #lambv = float(sys.argv[4].replace('regv',''))
        exp_type = int(sys.argv[2])
        lamb = float(sys.argv[3].replace('reg',''))
        eps = float(sys.argv[4].replace('eps',''))
        K = int(sys.argv[5].replace('K',''))
        F = int(sys.argv[6].replace('F',''))
        ite = int(sys.argv[7].replace('iter',''))

        init_conifg(dt, lamb, exp_type, eps, K, F)

        exp_id = int(time.time())
        solver = 'acc'
        if exp_type == 1:
            #ite = 3000
            run_start = time.time()
            split_num = 2
            t_dir = 'data/%s/exp_split/%s/' % (dt, split_num)

            train_filename = t_dir + '%s_train_%s.txt' % (rating_filename, split_num)
            test_filename = t_dir + '%s_test_%s.txt' % (rating_filename, split_num)

            rmses, maes = run(split_num, t_dir, lamb, K, eps, ite, solver, train_filename, test_filename, F)
            cost = (time.time() - run_start) / 3600.0
            logger.info('**********fm_with_group_lasso finish, run once, cost %.2f hours*******\nconfig: (K, reg, eps, ites, solver)=(%s, %s, %s, %s, %s), rmses: %s, maes: %s\navg rmse=%s, avg mae=%s\n***************', cost, K, lamb, eps, ite, solver, rmses[-5:], maes[-5:], np.mean(rmses[-5:]), np.mean(maes[-5:]))
        elif int(sys.argv[2]) == 2:
            run_5_validation(lamb, K, eps, ite, solver, F)
