#coding=utf8
'''
    solve the matrix factorization by block gradient descent, which can be applied to large scale datasets
'''
import time
import ctypes
import itertools
import sys
import logging

import numpy as np
from scipy.sparse import csr_matrix as cm
from numpy.linalg import norm
from numpy.linalg import svd
from numpy import power

import matplotlib.pyplot as plt

from logging_util import init_logger


def print_cost(func):
    def wrapper(*args, **kwargs):
        t = time.time()
        res = func(*args, **kwargs)
        print '%s: %.1fs' % (func.__name__, time.time() - t)
        return res
    return wrapper

class MF_BGD(object):

    def __init__(self, data=None, train_data=[], test_data=[], max_iter=500, K=10, lamb=0.1, eps=0.01, silent_run=False, save_uv=False, call_logger=None):
        if call_logger:
            global logger
            logger = call_logger
        self.K = K
        self.lamb = lamb
        self.eps = eps
        self.ite = max_iter
        self.tol = 1e-8
        self.train_ratio = 0.8
        self.silent_run=silent_run
        self.save_uv = save_uv
        if data is None:
            self.filename = 'data/ml-1m-rating.txt'
            self.load_data()
        else:
            self.data = data
            self.train_data = train_data
            self.train_num = len(self.train_data)
            self.test_data = test_data
            self.test_num = len(self.test_data)
        self.load_lib()

    def load_lib(self):
        part_dot_lib = ctypes.cdll.LoadLibrary('./partXY.so')
        set_val_lib = ctypes.cdll.LoadLibrary('./setVal.so')
        self.part_dot = part_dot_lib.partXY
        self.set_val = set_val_lib.setVal

    def split_data(self):
        rand_inds = np.random.permutation(self.obs_num)
        self.train_num = int(self.obs_num * self.train_ratio)

        self.train_data = self.data[rand_inds[:self.train_num]]
        self.test_data = self.data[rand_inds[self.train_num:]]
        self.test_num = len(self.test_data)
        del rand_inds

    def load_data(self):
        self.data = np.loadtxt(self.filename, dtype=np.float64)
        #self.data[:,2] -= self.data[:,2].mean()
        #self.data[:,2] /= self.data[:,2].std()
        self.obs_num = len(self.data)
        self.split_data()

    def get_obs_inds(self):
        return self.train_data[:,0].astype(int), self.train_data[:,1].astype(int)

    def part_uv(self, U, V, rows, cols, k):
        num = len(rows)
        output = np.zeros((num,1), dtype=np.float64)

        up = U.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        vp = V.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        op = output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        rsp = rows.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
        csp = cols.ctypes.data_as(ctypes.POINTER(ctypes.c_long))

        nc = ctypes.c_int(num)
        rc = ctypes.c_int(k)
        self.part_dot(up, vp, rsp, csp, op, nc, rc)
        return output

    def p_omega(self, mat, rows, cols):
        mat_t = mat.copy()
        mat_t[rows, cols] = 0.0
        return mat - mat_t

    def cal_omega(self, omega, U, V, rows, cols, bias, obs):
        puv = self.part_uv(U, V, rows, cols, self.K)
        puv = obs - puv -  bias
        puvp = puv.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        odp = omega.data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        nc = ctypes.c_int(self.train_num)
        self.set_val(puvp, odp, nc)

    def obj(self, U, V, omega):
        return 1.0 / 2 * power(norm(omega.data),2) + self.lamb / 2.0 * (power(norm(U,'fro'),2) + power(norm(V,'fro'),2))

    def train_rmse(self, U, V, bias, omega):
        return np.sqrt(power(norm(omega.data),2) / self.train_num)

    def get_grad(self, omega, U, V):
        du = -omega.dot(V) + self.lamb * U
        dv = -omega.T.dot(U) + self.lamb * V
        return du, dv

    def run(self):
        logger.info('MF running: parras: K=%s, reg=%s, lr=%s, silent_run=%s', self.K, self.lamb, self.eps, self.silent_run)
        X = cm((self.data[:,2], (self.data[:,0], self.data[:,1]))) #index starting from 0
        M, N = X.shape
        omega = cm((self.train_data[:,2], (self.train_data[:,0], self.train_data[:,1])), shape=(M,N)) #index starting from 0
        if len(self.test_data):
            trows, tcols = self.test_data[:,0].astype(np.int32), self.test_data[:,1].astype(np.int32)

        U = np.random.rand(M, self.K) * 0.0002
        V = np.random.rand(N, self.K) * 0.0002
        bias = self.train_data[:,2].mean()# in reality, bias can also be updated, modified later
        #bias = 0.0
        eps_1 = eps_2 = self.eps

        rows, cols = omega.tocoo().row.astype(np.int32), omega.tocoo().col.astype(np.int32)
        obs = omega.copy().data.astype(np.float64).reshape(self.train_num, 1)
        self.cal_omega(omega, U, V, rows, cols, bias, obs)

        objs_1 = [self.obj(U, V, omega)]
        objs_2 = []
        trmses = []
        rmses, maes, costs, acu_cost = [], [], [], []

        run_start = time.time()
        for rnd in range(0, self.ite):
            start = time.time()
            self.cal_omega(omega, U, V, rows, cols, bias, obs)
            #grad_bias = -omega + self.lamb * bias
            #bias = bias - 1.0/eps_1 * grad_bias
            du, dv = self.get_grad(omega, U, V)
            l_omega = omega.copy()
            for t1 in range(0, 20):
                #line search
                LU = U - 1.0/eps_1 * du
                LV = V - 1.0/eps_1 * dv
                self.cal_omega(l_omega, LU, LV, rows, cols, bias, obs)
                l_obj = self.obj(LU, LV, l_omega)
                if l_obj < objs_1[rnd]:
                    U, V = LU, LV
                    eps_1 *= 0.95
                    objs_1.append(l_obj)
                    trmses.append(self.train_rmse(U, V, bias,l_omega))
                    break
                else:
                    eps_1 *= 1.5

            if t1 == 19:
                break

            lrate = (objs_1[rnd] - objs_1[rnd+1]) / objs_1[rnd]

            end = time.time()
            costs.append(round(end-start, 1))
            acu_cost.append(int(end-run_start))

            if len(self.test_data):
                preds = self.part_uv(U, V, trows, tcols, self.K)
                rmses.append(self.cal_rmse(preds))
                maes.append(self.cal_mae(preds))
                if not self.silent_run:
                    logger.info('iter=%s, obj=%.4f(%.2f%%), ls:((%.4f, %s), (%.4f, %s)), train_rmse=%.4f,rmse=%.4f, mae=%.4f, time:%.1fs', rnd, objs_1[rnd], lrate * 100, eps_1, t1, eps_1, t1, trmses[rnd], rmses[rnd], maes[rnd], end-start)
            else:
                logger.info('iter=%s, obj=%.4f(%.2f%%), ls:((%.4f, %s), (%.4f, %s)), train_rmse=%.4f, time:%.1fs', rnd, objs_1[rnd], lrate * 100, eps_1, t1, eps_1, t1, trmses[rnd], end-start)

            if abs(lrate) < self.tol:
                #import pdb;pdb.set_trace()
                break

            if objs_1[rnd] < self.tol:
                break

        self.rmses = rmses if rmses else 99.0
        self.maes = maes if maes else 99.0
        if self.save_uv:
            np.savetxt(dir_+'mf_features/ratings_only/U_K%s.res' % self.K, U)
            np.savetxt(dir_+'mf_features/ratings_only/V_K%s.res' % self.K, V)
        return U, V

    def get_test_rmse(self):
        return np.mean(self.rmses[-5:])

    def get_test_mae(self):
        return np.mean(self.maes[-5:])

    def cal_rmse(self, preds):
        #user or item not occured in train dataset are set to 3 as default
        # beyond the 1,5 also need to set to 1 or 5
        delta = preds - self.test_data[:,2].reshape(preds.shape)
        rmse = np.sqrt(np.square(delta).sum() / self.test_num)
        return rmse

    def cal_mae(self, preds):
        delta = preds - self.test_data[:,2].reshape(preds.shape)
        mae = np.abs(delta).sum() / self.test_num
        return mae

def run_basedline(filename, K, eps, lamb, max_iter, silent_run=True):
    '''
        mf as baseline
    '''
    start_time = time.time()
    ratings = np.loadtxt(filename,dtype=np.float64)
    uids = set([int(r) for r in ratings[:,0]])
    bids = set([int(r) for r in ratings[:,1]])
    uid2iid = {v:k for k,v in enumerate(uids)}
    bid2iid = {v:k for k,v in enumerate(bids)}
    ratings[:,0] = [uid2iid[int(r)] for r in ratings[:,0]]
    ratings[:,1] = [bid2iid[int(r)] for r in ratings[:,1]]
    print 'finish loading data, cost %.2f seconds, users: %s, items: %s, obs: %s, density=%.4f' % (time.time() - start_time, len(uids), len(bids), len(ratings), len(ratings) * 1.0 / len(uids) / len(bids))

    rand_inds = np.random.permutation(ratings.shape[0])
    train_num = int(ratings.shape[0] * 0.8)
    train_data = ratings[rand_inds[:train_num]]
    test_data = ratings[rand_inds[train_num:]]
    mf = MF_BGD(data=ratings, train_data=train_data, test_data=test_data, max_iter=max_iter, K=K, eps=eps, lamb=lamb,silent_run=silent_run)
    mf.run()
    return mf.get_test_rmse()

def run_5_validations(rating_filename, K, eps, lamb, max_iter, silent_run=True):
    '''
        mf as baseline
    '''
    exp_id = int(time.time())
    logger.info('start run_5_validations, ratings_filename=%s, K=%s,eps=%s,reg=%s,max_iter=%s', rating_filename, K,eps,lamb,max_iter)
    ratings = np.loadtxt(rating_filename,dtype=np.float64)
    rating_filename = rating_filename.split('/')[-1].replace('.txt', '')
    uids = set([int(r) for r in ratings[:,0]])
    bids = set([int(r) for r in ratings[:,1]])
    uid2iid = {v:k for k,v in enumerate(uids)}
    bid2iid = {v:k for k,v in enumerate(bids)}
    ratings[:,0] = [uid2iid[int(r)] for r in ratings[:,0]]
    ratings[:,1] = [bid2iid[int(r)] for r in ratings[:,1]]

    def get_triplets(filename):
        triplets = np.loadtxt(filename, dtype=np.float64)

        triplets[:,0] = [uid2iid[int(r)] for r in triplets[:,0]]
        triplets[:,1] = [bid2iid[int(r)] for r in triplets[:,1]]
        return triplets

    exp_rmses, exp_maes = [], []

    val_start = time.time()
    for rnd in xrange(5):
        start_time = time.time()

        train_filename = dir_ + 'exp_split/%s/%s_train_%s.txt' % (rnd+1, rating_filename, rnd+1)
        test_filename = dir_ + 'exp_split/%s/%s_test_%s.txt' % (rnd+1, rating_filename, rnd+1)
        logger.info('start validation %s, train_filename=%s, test_filename=%s', rnd+1, train_filename, test_filename)

        train_data = get_triplets(train_filename)
        test_data = get_triplets(test_filename)

        mf = MF_BGD(data=ratings, train_data=train_data, test_data=test_data, max_iter=max_iter, K=K, eps=eps, lamb=lamb,silent_run=silent_run)
        mf.run()
        exp_rmses.append(mf.get_test_rmse())
        exp_maes.append(mf.get_test_mae())
        logger.info('finish validation %s, cost %.2f minutes, rmse=%.4f, mae=%.4f', rnd+1, (time.time() - start_time) / 60.0, exp_rmses[rnd], exp_maes[rnd])
    val_end = time.time()
    logger.info('**********finish 5 validations,rating from %s, cost %.2f minutes!K=%s,eps=%s,lamb=%s,max_iter=%s******\n*******exp_rmses=%s, exp_rmses=%s******\n*****avg: rmse=%.4f,mae=%.4f****', rating_filename, (val_end - val_start) / 60.0, K, eps, lamb, max_iter, exp_rmses, exp_maes, np.mean(exp_rmses), np.mean(exp_maes))

def grid_search(filename):
    #filenames = ['ratings_filter5.txt',]
    print 'grid search for %s' % filename
    Ks = [10,20,30,40,50]
    epss = [0.001, 0.01, 0.1, 1, 10, 100]
    lambs = [0.01,0.1,1,5,10, 100]
    res = []
    for K, eps, lamb in itertools.product(Ks, epss, lambs):
        rmse = run_basedline(filename, K, eps, lamb)
        res.append((filename, K, eps,lamb, rmse))
    res = sorted(res, key=lambda d:d[-1], reverse=False)
    fw = open('data/yelp/samples/grid_res_%s' % filename, 'w+')
    fw.write('\n'.join(['%s\t%s\t%s\t%s\t%s' % (f,k,e,l,r) for f,k,e,l,r in res]))
    fw.close()

if __name__ == '__main__':
    if len(sys.argv) == 3:
        dt = sys.argv[1]

        global logger
        global dir_
        dir_ = 'data/%s/' % dt
        exp_id = int(time.time())
        log_filename = 'log/%s_mf.log' % dt
        logger = init_logger('exp_%s' % str(exp_id), log_filename, logging.INFO, False)
        if int(sys.argv[2]) == 1:
            filename = dir_ + 'ratings.txt'
            K = 10
            eps = 10
            lamb = 1
            max_iter = 500
            exp_rmses = []
            for i in range(1):
                exp_rmses.append(run_basedline(filename, K, eps, lamb, max_iter, silent_run=False))
            print 'K=%s,eps=%s,lamb=%s,max_iter=%s, exp_rmses=%s,avg_rmse=%s' % (K,eps,lamb,max_iter, exp_rmses, np.mean(exp_rmses))
        elif int(sys.argv[2]) == 2:
            filename = dir_ + 'ratings.txt'
            K = 10
            eps = 10
            lamb = 10
            max_iter = 1000
            run_5_validations(filename, K, eps, lamb, max_iter, silent_run=False)
        elif int(sys.argv[2]) == 3:
            filename = 'UBNUI.res'
            grid_search(filename)
