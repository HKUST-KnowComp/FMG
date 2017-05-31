#coding=utf8
'''
    standard fm, i.e. poly regression with anova kernel
    regularization is group lasso
'''
import time
import logging

import numpy as np
from numpy.linalg import norm

from exp_util import cal_rmse, cal_mae

stf = lambda eta, nw: 1.0 - eta / nw if eta < nw else 0.0#soft threshold function

class FMAKGL(object):

    def __init__(self, config, data_loader):
        self.config = config
        self.train_X, self.train_Y, self.test_X, self.test_Y = data_loader.get_exp_data()
        self._init_config()

    def _init_config(self):
        self.exp_id = self.config.get('exp_id')
        self.N = self.config.get('N')
        self.K = self.config.get('K')
        self.L = self.config.get('L')
        self.F = self.config.get('F')
        self.initial = self.config.get('initial')
        self.reg_W = self.config.get('reg_W')
        self.reg_P = self.config.get('reg_P')
        self.max_iters = self.config.get('max_iters')
        self.ln = self.config.get('ln')
        self.eps = self.config.get('eps')
        self.eta = self.config.get('eta')
        self.solver = self.config.get('solver')
        self.bias_eta = self.config.get('eta')
        self.bias = np.mean(self.train_Y)
        #better to add log information for the configs

        self.M = self.train_X.shape[0]

    def _prox_op(self, eta, G, g_inds):
        for i in range(len(g_inds)):
            G[g_inds[i]] = stf(eta, norm(G[g_inds[i]])) * G[g_inds[i]]
        return G

    def _group_lasso(self, G, g_inds):
        res = 0.0
        for i in range(g_inds.shape[0]):
            res += norm(G[g_inds[i]])
        return res

    def _obj(self, err, W, P):
        part1 = np.power(err, 2).sum() / self.M
        part2 = self.reg_W * self._group_lasso(W, self.gw_inds)
        part3 = self.reg_P * self._group_lasso(P.flatten(), self.gp_inds)
        logging.debug('obj detail, part1=%s, part2=%s, part3=%s', part1, part2, part3)
        return part1 + part2 + part3

    def _cal_err(self, WX, XP, XSPS, Y):
        Y_t = self.bias + WX + 0.5 * (np.square(XP) - XSPS).sum(axis=1)
        return Y_t - Y

    def _get_XC_prods(self, X, W, P):
        WX = np.dot(W, X.T)
        XP = np.dot(X, P)
        XSPS = np.dot(np.square(X), np.square(P))
        return WX, XP, XSPS

    def get_eval_res(self):
        return self.rmses, self.maes

    def train(self):


        W = np.random.rand(self.N) * self.initial # 1 by N
        P = np.random.rand(self.N, self.K) * self.initial# N by K
        self.gw_inds = np.arange(self.N).reshape(2 * self.L, self.F)
        self.gp_inds = np.arange(self.N * self.K).reshape(2 * self.L, self.F * self.K)

        if self.solver == 'PG':
            self._block_proximal_gradient_descent(W, P)
        elif self.solver == 'mAPG':
            self._block_mono_acc_proximal_gradient_descent(W, P)
        elif self.solver == 'nmAPG':
            self._block_nonmono_acc_proximal_gradient_descent(W, P)

    def _block_proximal_gradient_descent(self, W, P):
        WX, XP, XSPS = self._get_XC_prods(self.train_X, W, P)
        err = self._cal_err(WX, XP, XSPS, self.train_Y)
        objs = [self._obj(err, W, P)]

        WtX, tXP, tXSPS = self._get_XC_prods(self.test_X, W, P)
        test_err = self._cal_err(WtX, tXP, tXSPS, self.test_Y)
        rmses = [cal_rmse(test_err)]
        maes = [cal_mae(test_err)]

        start = time.time()

        eta = self.eta
        for t in range(self.max_iters):
            start = time.time()

            l_obj, eta, lt, W, P = self._get_updated_paras(eta, W, P)

            if lt == self.ln:
                logging.info('!!!stopped by line_search, lt=%s!!!', lt)
                break

            objs.append(l_obj)

            WtX, tXP, tXSPS = self._get_XC_prods(self.test_X, W, P)
            test_err = self._cal_err(WtX, tXP, tXSPS, self.test_Y)
            rmses.append(cal_rmse(test_err))
            maes.append(cal_mae(test_err))
            end = time.time()

            dr = abs(objs[t] - objs[t+1]) / objs[t]
            logging.info('exp_id=%s, iter=%s, lt,eta,dr=(%s,%s, %.7f), obj=%.5f, rmse=%.5f, mae=%.5f, cost=%.2f seconds', self.exp_id, t, lt, eta, dr, objs[t], rmses[t], maes[t], (end - start))
            if  dr < self.eps:
                logging.info('*************stopping criterion satisfied*********')
                break

        logging.info('train process finished, total iters=%s', t+1)
        self.rmses, self.maes = rmses, maes
        self._save_paras(W, P)

    def _block_mono_acc_proximal_gradient_descent(self, W, P):
        '''
            monotone accelerated pg
        '''
        logging.info('**********start solving by _block_mono_acc_proximal_gradient_descent****************')
        WX, XP, XSPS = self._get_XC_prods(self.train_X, W, P)
        err = self._cal_err(WX, XP, XSPS, self.train_Y)
        objs = [self._obj(err, W, P)]

        WtX, tXP, tXSPS = self._get_XC_prods(self.test_X, W, P)
        test_err = self._cal_err(WtX, tXP, tXSPS, self.test_Y)
        rmses = [cal_rmse(test_err)]
        maes = [cal_mae(test_err)]

        start = time.time()

        A = np.hstack((W.reshape(-1,1), P))
        A0, A1, C1 = A.copy(), A.copy(), A.copy()
        r0, r1 = 0, 1

        eta = self.eta
        XS = np.square(self.train_X)
        for t in range(self.max_iters):
            start = time.time()

            v_obj, v_eta, v_lt, vW, vP = self._get_updated_paras(eta, W, P)

            B = A1 + r0/r1 * (C1 - A1) + (r0 - 1)/r1 * (A1 - A0)
            W, P = B[:,0].flatten(), B[:,1:]
            y_obj, y_eta, y_lt, yW, yP = self._get_updated_paras(eta, W, P)
            C1 = np.hstack((yW.reshape(-1,1), yP))

            if v_obj > y_obj:
                objs.append(y_obj)
                lt = y_lt
                eta = y_eta
                W, P = yW, yP
            else:
                objs.append(v_obj)
                lt = v_lt
                eta = v_eta
                W, P = vW, vP

            if lt == self.ln:
                logging.info('!!!stopped by line_search, lt=%s!!!', lt)
                break

            A0 = A1
            A1 = np.hstack((W.reshape(-1,1), P))
            r1 = (np.sqrt(4 * pow(r0, 2) + 1) + 1) / 2.0

            WtX, tXP, tXSPS = self._get_XC_prods(self.test_X, W, P)
            test_err = self._cal_err(WtX, tXP, tXSPS, self.test_Y)
            rmses.append(cal_rmse(test_err))
            maes.append(cal_mae(test_err))
            end = time.time()

            dr = abs(objs[t] - objs[t+1]) / objs[t]
            logging.info('exp_id=%s, iter=%s, lt,eta,dr=(%s,%s, %.7f), obj=%.5f, rmse=%.5f, mae=%.5f, cost=%.2f seconds', self.exp_id, t, lt, eta, dr, objs[t], rmses[t], maes[t], (end - start))
            if  dr < self.eps:
                logging.info('*************stopping criterion satisfied*********')
                break

        logging.info('train process finished, total iters=%s', t+1)
        self.rmses, self.maes = rmses, maes
        self._save_paras(W, P)

    def _block_nonmono_acc_proximal_gradient_descent(self, W, P):
        '''
            non-monotone accelerated pg
        '''
        logging.info('start solving by _block_nonmono_acc_proximal_gradient_descent')
        WX, XP, XSPS = self._get_XC_prods(self.train_X, W, P)
        err = self._cal_err(WX, XP, XSPS, self.train_Y)
        objs = [None] * (self.max_iters + 1)
        objs[0]= self._obj(err, W, P)

        WtX, tXP, tXSPS = self._get_XC_prods(self.test_X, W, P)
        test_err = self._cal_err(WtX, tXP, tXSPS, self.test_Y)
        rmses = [cal_rmse(test_err)]
        maes = [cal_mae(test_err)]

        start = time.time()

        A = np.hstack((W.reshape(-1,1), P))
        A0, A1, C1 = A.copy(), A.copy(), A.copy()
        c = objs[0]
        r0, r1, q, qeta = 0.0, 1.0, 1.0, 0.5
        eta1 = eta2 = self.eta
        lt1, lt2 = 0, 0

        XS = np.square(self.train_X)
        for t in range(self.max_iters):
            start = time.time()
            self._update_bias(W, P)

            B = A1 + r0/r1 * (C1 - A1) + (r0 - 1)/r1 * (A1 - A0)
            W, P = B[:,0].flatten(), B[:,1:]
            y_obj, y_eta, y_lt, yW, yP = self._get_updated_paras(eta1, W, P)
            lt1, eta1 = y_lt, y_eta
            C1 = np.hstack((yW.reshape(-1,1), yP))

            if y_obj < c:
                objs[t+1] = y_obj
                W, P = yW, yP
            else:
                W, P = A1[:,0].flatten(), A1[:,1:]
                v_obj, v_eta, v_lt, vW, vP = self._get_updated_paras(eta2, W, P)
                lt2, eta2 = v_lt, v_eta

                if y_obj < v_obj:
                    objs[t+1] = y_obj
                    W, P = yW, yP
                else:
                    objs[t+1] = v_obj
                    W, P = vW, vP

            if lt1 == self.ln or lt2 == self.ln:
                logging.info('!!!stopped by line_search, lt1=%s, lt2=%s!!!', lt1, lt2)
                break

            A0 = A1
            A1 = np.hstack((W.reshape(-1,1), P))

            r0 = r1
            r1 = (np.sqrt(4 * pow(r0, 2) + 1) + 1) / 2.0
            tq = qeta * q + 1.0
            c = (qeta * q * c + objs[t+1]) / tq
            q = tq

            WtX, tXP, tXSPS = self._get_XC_prods(self.test_X, W, P)
            test_err = self._cal_err(WtX, tXP, tXSPS, self.test_Y)
            rmses.append(cal_rmse(test_err))
            maes.append(cal_mae(test_err))
            end = time.time()

            dr = abs(objs[t] - objs[t+1]) / objs[t]
            logging.info('exp_id=%s, iter=%s, (lt1,eta1,lt2,eta2)=(%s,%s,%s,%s), obj=%.5f(%.8f), rmse=%.5f, mae=%.5f, cost=%.2f seconds', self.exp_id, t, lt1, eta1, lt2, eta2, objs[t], dr, rmses[t], maes[t], (end - start))
            if  dr < self.eps:
                logging.info('*************stopping criterion satisfied*********')
                break

        logging.info('train process finished, total iters=%s', t+1)
        self.rmses, self.maes = rmses, maes
        self._save_paras(W, P)

    def _get_updated_paras(self, eta, W, P):

        WX, XP, XSPS = self._get_XC_prods(self.train_X, W, P)
        err = self._cal_err(WX, XP, XSPS, self.train_Y)
        obj_t = self._obj(err, W, P)

        #cal gradients
        grad_W = 2.0 / self.M * np.dot(err, self.train_X)#element-wise correspondence

        XS = np.square(self.train_X)
        grad_P = np.zeros(P.shape)
        for f in range(self.K):
            grad_P[:,f] = 2.0 / self.M * np.dot(err, np.multiply(self.train_X, XP[:,f].reshape(-1,1).repeat(self.N, axis=1)) - np.multiply(P[:,f].reshape(1, -1).repeat(self.M, axis=0), XS))

        l_obj, eta, lt, W, P = self._line_search(obj_t, eta, W, P, grad_W, grad_P)

        return l_obj, eta, lt, W, P

    def _update_bias(self, W, P):
        WX, XP, XSPS = self._get_XC_prods(self.train_X, W, P)
        err = self._cal_err(WX, XP, XSPS, self.train_Y)
        self.bias -= self.bias_eta * 2.0 / self.M * err.sum()

    def _line_search(self, obj_v, eta, W, P, grad_W, grad_P):
        for lt in range(self.ln+1):
            lW = W - eta * grad_W
            lW = self._prox_op(eta * self.reg_W, lW, self.gw_inds)

            lP = P - eta * grad_P
            lP = self._prox_op(eta * self.reg_P, lP.flatten(), self.gp_inds)
            lP = lP.reshape(P.shape)

            lWX, XlP, XSlPS = self._get_XC_prods(self.train_X, lW, lP)
            l_err = self._cal_err(lWX, XlP, XSlPS, self.train_Y)
            l_obj = self._obj(l_err, lW, lP)

            if l_obj < obj_v:
                eta *= 1.1
                W, P = lW, lP
                break
            else:
                eta *= 0.7
        return l_obj, eta, lt, W, P

    def _save_paras(self, W, P):
        split_num = self.config['sn']
        dt = self.config.get('dt')
        W_wfilename = 'fm_res/%s_split%s_W_%s_exp%s.txt' % (dt, split_num, self.reg_W, self.exp_id)
        np.savetxt(W_wfilename, W)
        P_wfilename = 'fm_res/%s_split%s_P_%s_exp%s.txt' % (dt, split_num, self.reg_P, self.exp_id)
        np.savetxt(P_wfilename, P)
        logging.info('W and P saved in %s and %s', W_wfilename, P_wfilename)

