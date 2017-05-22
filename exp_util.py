#coding=utf8
'''
    utils for experiments
'''
import numpy as np

def cal_rmse(test_err):
    num = test_err.size
    rmse = np.sqrt(np.square(test_err).sum() / num)
    return rmse

def cal_mae(test_err):
    num = test_err.size
    mae = np.abs(test_err).sum() / num
    return mae

