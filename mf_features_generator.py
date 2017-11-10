#coding=utf8
'''
    generate MF features from the meta-structure similarity
'''

import sys
import time
import logging
import numpy as np
#from numba import jit

from mf import MF_BGD as MF
from utils import reverse_map

from logging_util import init_logger

topK = 500

def run(path_str, comb='', K=10):
    if path_str in ['ratings_only']:
        use_topK = False
    else:
        use_topK = True

    sim_filename = dir_ + 'sim_res/path_count/%s.res' % path_str
    if path_str == 'ratings_only':
        sim_filename = dir_ + 'ratings.txt'
    if use_topK:
        sim_filename = dir_ + 'sim_res/path_count/%s_top%s.res' % (path_str, topK)
    if comb:
        sim_filename = dir_ + 'sim_res/path_count/combs/%s_%s_top%s.res' % (path_str, comb, topK)
    start_time = time.time()
    data = np.loadtxt(sim_filename)
    uids = set(data[:,0].flatten())
    bids = set(data[:,1].flatten())
    uid2ind = {int(v):k for k,v in enumerate(uids)}
    ind2uid = reverse_map(uid2ind)
    bid2ind = {int(v):k for k,v in enumerate(bids)}
    ind2bid = reverse_map(bid2ind)

    data[:,0] = [uid2ind[int(r)] for r in data[:,0]]
    data[:,1] = [bid2ind[int(r)] for r in data[:,1]]

    print 'finish load data from %s, cost %.2f seconds, users: %s, items=%s' % (sim_filename, time.time() - start_time, len(uids), len(bids))

    eps, lamb, iters = 10, 10, 500
    print 'start generate mf features, (K, eps, reg, iters) = (%s, %s, %s, %s)' % (K, eps, lamb, iters)
    mf = MF(data=data, train_data=data, test_data=[], K=K, eps=eps, lamb=lamb, max_iter=iters, call_logger=logger)
    U,V = mf.run()
    start_time = time.time()
    wfilename = dir_ + 'mf_features/path_count/%s_user.dat' % (path_str)
    if use_topK:
        #wfilename = dir_ + 'mf_features/path_count/ranks/%s_top%s_F%s_user.dat' % (path_str, topK, K)
        wfilename = dir_ + 'mf_features/path_count/%s_top%s_user.dat' % (path_str, topK)
    if comb:
        wfilename = dir_ + 'mf_features/path_count/combs/%s_%s_top%s_user.res' % (path_str, comb, topK)

    fw = open(wfilename, 'w+')
    res = []
    for ind, fs in enumerate(U):
        row = []
        row.append(ind2uid[ind])
        row.extend(fs.flatten())
        res.append('\t'.join([str(t) for t in row]))

    fw.write('\n'.join(res))
    fw.close()
    print 'User-Features: %s saved in %s, cost %.2f seconds' % (U.shape, wfilename, time.time() - start_time)

    start_time = time.time()
    wfilename = dir_ + 'mf_features/path_count/%s_item.dat' % (path_str)
    if use_topK:
        wfilename = dir_ + 'mf_features/path_count/%s_top%s_item.dat' % (path_str, topK)
        #wfilename = dir_ + 'mf_features/path_count/ranks/%s_top%s_F%s_item.dat' % (path_str, topK, K)
    if comb:
        wfilename = dir_ + 'mf_features/path_count/combs/%s_%s_top%s_item.res' % (path_str, comb, topK)

    fw = open(wfilename, 'w+')
    res = []
    for ind, fs in enumerate(V):
        row = []
        row.append(ind2bid[ind])
        row.extend(fs.flatten())
        res.append('\t'.join([str(t) for t in row]))

    fw.write('\n'.join(res))
    fw.close()
    print 'Item-Features: %s  saved in %s, cost %.2f seconds' % (V.shape, wfilename, time.time() - start_time)

def run_all_yelp():
    for path_str in ['UPBCatB','UPBCityB', 'UPBStateB', 'UPBStarsB']:
        run(path_str)
    for path_str in ['UPBUB', 'UNBUB', 'URPARUB', 'URNARUB', 'UUB']:
        run(path_str)
    for path_str in ['URPSRUB', 'URNSRUB']:
        run(path_str)
    for path_str in ['ratings_only']:
        run(path_str)

def run_all_yelp_by_rank():
    for K in [2,3,5,20,30,40,50,100]:
        start = time.time()
        print 'process rank ', K
        for path_str in ['UPBCatB','UPBCityB', 'UPBStateB', 'UPBStarsB']:
            run(path_str, K=K)
        for path_str in ['UPBUB', 'UNBUB', 'URPARUB', 'URNARUB', 'UUB']:
            run(path_str, K=K)
        for path_str in ['URPSRUB', 'URNSRUB']:
            run(path_str, K=K)
        for path_str in ['ratings_only']:
            run(path_str, K=K)
        print 'finish processing rank %s, cost %.2fm ' % (K, (time.time() - start) / 60.0)

def run_all_amazon_by_rank():
    for K in [2,3,5,20,30,40,50,100]:
        start = time.time()
        print 'process rank ', K
        for path_str in ['UPBCatB','UPBBrandB']:
            run(path_str, K=K)
        for path_str in ['UPBUB', 'UNBUB', 'URPARUB', 'URNARUB']:
            run(path_str, K=K)
        for path_str in ['URPSRUB', 'URNSRUB']:
            run(path_str, K=K)
        #for path_str in ['ratings_only']:
        #    run(path_str, K=K)
        print 'finish processing rank %s, cost %.2fm ' % (K, (time.time() - start) / 60.0)

def run_cikm_yelp():
    #for path_str in ['UPBUB', 'UNBUB', 'UPBCatBUB', 'UNBCatBUB','UPBCityBUB', 'UNBCityBUB', 'UUB', 'UCompUB']:
    for path_str in ['ratings_only']:
        run(path_str)

def run_all_douban():
    for path_str in ['UBUB', 'UGUB', 'UBDBUB', 'UBABUB', 'UBTBUB', 'ratings_only']:
    #for path_str in ['ratings_only']:
        run(path_str)

def run_all_amazon_200k(ratings_only=False):
    if ratings_only:
        run('ratings_only')
    else:
        for path_str in ['UPBCatB','UPBBrandB']:
            run(path_str)
        for path_str in ['UPBUB', 'UNBUB', 'URPARUB', 'URNARUB']:
            run(path_str)
        for path_str in ['URPSRUB', 'URNSRUB']:
            run(path_str)

def run_amazon_200k_combs():
    run_start = time.time()
    path_strs = ['UBUB', 'URARUB']
    #path_strs = ['URARUB']
    combs = ['PPP', 'NNP', 'PPN', 'NNN', 'PNP', 'NPP', 'PNN', 'NPN']
    cnt = 1
    for path_str in path_strs:
        for comb in combs:
            print 'start processing %s_%s, cnt=%s' % (path_str, comb, cnt)
            cnt += 1
            run(path_str, comb)
            print 'finish processing %s_%s, cnt=%s' % (path_str, comb, cnt)

if __name__ == '__main__':
    global dir_
    if len(sys.argv) == 4:
        dt = sys.argv[1]
        path_str = sys.argv[2]
        split_num = sys.argv[3]
        dir_ = 'data/%s/exp_split/%s/' % (dt, split_num)
        log_filename = 'log/%s_mf_feature_geneartion_%s_split%s.log' % (dt, path_str, split_num)
        exp_id = int(time.time())
        logger = init_logger('exp_%s' % exp_id, log_filename, logging.INFO, False)
        print 'data: %s, path_str: %s' % (dir_, path_str)
        logger.info('data: %s, path_str: %s', dir_, path_str)
        if path_str == 'all':
            if 'yelp' in dt:
                run_all_yelp()
            elif 'amazon' in dt:
                run_all_amazon_200k()
            elif 'douban' in dt:
                run_all_douban()
        elif path_str == 'all-rank':
            if 'yelp' in dt:
                run_all_yelp_by_rank()
            elif 'amazon' in dt:
                run_all_amazon_by_rank()
        elif path_str == 'cikm':
            dir_ = 'data/cikm/yelp/exp_split/%s/' % (split_num)
            print 'data: %s, path_str: %s' % (dir_, path_str)
            run_cikm_yelp()
        elif path_str == 'comb':
            if 'yelp' in dt:
                pass
            elif 'amazon' in dt:
                run_amazon_200k_combs()
        else:
            run(path_str)
    else:
        print 'please speficy the data and path_str'
        sys.exit(0)

