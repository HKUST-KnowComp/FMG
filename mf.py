#coding=utf8
'''
    solve the matrix factorization by block gradient descent, which can be applied to large scale datasets
'''
import time
import ctypes
import logging

import numpy as np
from scipy.sparse import csr_matrix as cm
from numpy.linalg import norm
from numpy.linalg import svd
from numpy import power

import matplotlib.pyplot as plt

def print_cost(func):
    def wrapper(*args, **kwargs):
        t = time.time()
        res = func(*args, **kwargs)
        print '%s: %.1fs' % (func.__name__, time.time() - t)
        return res
    return wrapper

class MF_BGD(object):

    def __init__(self, data, train_data, test_data, **paras):
        self.K = paras['K']
        self.reg = paras['reg']
        self.eps = paras['eps']
        self.initial = paras['initial']
        self.ite = paras['max_iter']
        self.tol = paras['tol']
        self.data = data
        self.obs_num = len(self.data)
        self.train_data = train_data
        self.train_num = len(self.train_data)
        self.test_data = test_data
        self.test_num = len(self.test_data)
        self.load_lib()
        self.X = cm((self.data[:,2], (self.data[:,0], self.data[:,1]))) #index starting from 0
        self.M, self.N = self.X.shape
        logging.info('finish initiating the model, paras=%s', paras)

    def load_lib(self):
        part_dot_lib = ctypes.cdll.LoadLibrary('./partXY_blas.so')
        set_val_lib = ctypes.cdll.LoadLibrary('./setVal.so')
        self.part_dot = part_dot_lib.partXY
        self.set_val = set_val_lib.setVal

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
        return 1.0 / 2 * power(norm(omega.data),2) + self.reg / 2.0 * (power(norm(U,'fro'),2) + power(norm(V,'fro'),2))

    def train_rmse(self, U, V, bias, omega):
        return np.sqrt(power(norm(omega.data),2) / self.train_num)

    def get_grad(self, omega, U, V):
        du = -omega.dot(V) + self.reg * U
        dv = -omega.T.dot(U) + self.reg * V
        return du, dv

    def run(self):
        omega = cm((self.train_data[:,2], (self.train_data[:,0], self.train_data[:,1])), shape=(self.M, self.N)) #index starting from 0
        trows, tcols = self.test_data[:,0].astype(np.int32), self.test_data[:,1].astype(np.int32)
        ground = self.test_data[:,2]

        U = np.random.rand(self.M, self.K) * self.initial
        V = np.random.rand(self.N, self.K) * self.initial
        bias = np.mean(self.train_data[:,2])# in reality, bias can also be updated, modified later
        eps = self.eps

        rows, cols = omega.tocoo().row.astype(np.int32), omega.tocoo().col.astype(np.int32)
        obs = omega.copy().data.astype(np.float64).reshape(self.train_num, 1)
        self.cal_omega(omega, U, V, rows, cols, bias, obs)

        objs = [self.obj(U, V, omega)]
        trmses = []
        rmses, maes, costs, acu_cost = [], [], [], []

        run_start = time.time()
        for rnd in range(0, self.ite):
            start = time.time()
            self.cal_omega(omega, U, V, rows, cols, bias, obs)
            du, dv = self.get_grad(omega, U, V)
            l_omega = omega.copy()
            for t1 in range(0, 20):
                #line search
                LU = U - 1.0/eps * du
                LV = V - 1.0/eps * dv
                self.cal_omega(l_omega, LU, LV, rows, cols, bias, obs)
                l_obj = self.obj(LU, LV, l_omega)
                if l_obj < objs[rnd]:
                    U, V = LU, LV
                    eps *= 0.95
                    objs.append(l_obj)
                    trmses.append(self.train_rmse(U, V, bias, l_omega))
                    break
                else:
                    eps *= 1.5

            if t1 == 19:
                logging.info('*************stopped by linesearch**********')
                break

            lrate = (objs[rnd] - objs[rnd+1]) / objs[rnd]

            end = time.time()
            costs.append(round(end-start, 1))
            acu_cost.append(int(end-run_start))

            preds = self.part_uv(U, V, trows, tcols, self.K)
            preds += bias
            rmses.append(self.cal_rmse(preds, ground))
            maes.append(self.cal_mae(preds, ground))
            logging.info('iter=%s, obj=%.4f(%.7f), ls:((%.4f, %s), train_rmse=%.4f, rmse=%.4f, mae=%.4f, time:%.1fs', rnd+1, objs[rnd+1], lrate, eps, t1, trmses[rnd], rmses[rnd], maes[rnd], end-start)

            if abs(lrate) < self.tol:
                logging.info('stopped by tol, iter=%s', rnd+1)
                break

        self.U = U
        self.V = V
        self.bias = bias
        self.preds = preds
        return objs, trmses, rmses, maes, acu_cost

    def eval_in_mat_res(self):
        '''
            evaluate test obs where user and item occurs in train data
        '''
        train_uids = set(self.train_data[:,0].flatten().astype(int))
        train_bids = set(self.train_data[:,1].flatten().astype(int))
        in_mat_test_inds = [ind for ind, r in enumerate(self.test_data) if int(r[0]) in train_uids and int(r[1]) in train_bids]
        in_mat_test_preds = self.preds[in_mat_test_inds]
        in_mat_ground = self.test_data[in_mat_test_inds]
        self.in_mat_rmse = self.cal_rmse(in_mat_test_preds, in_mat_ground)
        self.in_mat_mae = self.cal_mae(in_mat_test_preds, in_mat_ground)

    def cal_rmse(self, preds, ground):
        delta = preds - ground.reshape(preds.shape)
        rmse = np.sqrt(np.square(delta).sum() / preds.size)
        return rmse

    def cal_mae(self, preds, ground):
        delta = preds - ground.reshape(preds.shape)
        mae = np.abs(delta).sum() / preds.size
        return mae

    def get_uv(self):
        return self.U, self.V

    def predict(self, uind, bind):
        '''
            Given the indices of user and item, predict the rating
        '''
        return np.dot(self.U[uind], self.V[bind]) + self.bias

    def get_preds(self):
        return self.preds

    def get_in_mat_eval(self):
        return self.in_mat_rmse, self.in_mat_mae
