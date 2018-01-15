#coding=utf8
'''
    all the detailed functions that calculate commuting matrix for every given path
'''
import numpy as np
import time

def cal_mat_uub(path_str, uu, uu_t, ub):
    print 'meta structure str is ', path_str
    t1 = time.time()
    UU = uu.dot(uu_t)
    t2 = time.time()
    print 'UU cost ', t2 - t1

    UUB = UU.dot(ub)
    t3 = time.time()
    print 'UUB cost ', t3 - t2
    return UUB

def cal_mat_urarub(path_str, ur, ur_t, ra, ra_t, ub):
    print 'meta structure str is ', path_str
    t1 = time.time()
    URA = ur.dot(ra)
    t2 = time.time()
    print 'UR cost ', t2 - t1

    URAR = URA.dot(ra_t)
    t3 = time.time()
    print 'URAR cost ', t3 - t2

    URARU = URAR.dot(ur_t)
    t4 = time.time()
    print 'URARU cost ', t4 - t3

    URARUB = URARU.dot(ub)
    t5 = time.time()
    print 'URARUB cost ', t5 - t4

    return URARUB

def cal_mat_bb(path_str, bo, bo_t):
    '''
        calculate the B-*-B
    '''
    print 'meta structure str is ', path_str
    t2 = time.time()
    BOB = bo.dot(bo_t)
    t3 = time.time()
    print 'BOB cost ', t3 - t2
    return BOB

def cal_mat_ubb(path_str, ub, bo, bo_t):
    '''
        calculate the U-B-*-B
    '''
    print 'meta structure str is ', path_str
    print ub.shape, bo.shape, bo_t.shape
    t2 = time.time()
    UBO = ub.dot(bo)
    t3 = time.time()
    print 'UBO(%s) cost %.2f seconds ' % (UBO.shape, t3 - t2)
    UBOB = UBO.dot(bo_t)
    t4 = time.time()
    print 'UBOB cost ', t4 - t3
    return UBOB

