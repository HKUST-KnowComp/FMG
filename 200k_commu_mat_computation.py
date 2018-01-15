#coding=utf8
'''
    calculate the similarity by commuting matrix baesd on meta-structure
'''
from numba import jit
import time
import sys
import os
import cPickle as pickle

import numpy as np
import bottleneck as bn
from scipy.sparse import csr_matrix as csr
from utils import reverse_map, generate_adj_mat, save_triplets

from cal_commuting_mat import *

def get_topK_items(comm_res, ind2uid, ind2bid, topK=1000):
    start = time.time()
    U, _ = comm_res.shape
    triplets = []
    for i in xrange(U):
        items = comm_res.getrow(i).toarray().flatten()
        cols = np.argpartition(-items, topK).flatten()[:topK]#descending order
        cols = [c for c in cols if items[c] > 0]#those less than 1000 non-zero entries, need to be removed zero ones
        triplets.extend([(ind2uid[i], ind2bid[c], items[c]) for c in cols])
    #logger.info('get topK items, total %s entries, cost %.2f seconds', len(triplets), time.time() - start)
    print 'get top %s items, total %s entries, cost %.2f seconds' % (topK, len(triplets), time.time() - start)
    return triplets

def batch_save_comm_res(path_str, wfilename, comm_res, ind2row, ind2col):
    coo = comm_res.tocoo(copy=False)
    step = 10000000
    N = len(coo.row) / step
    for i in range(N+1):
        start_time = time.time()
        triplets = []
        start = i * step
        end = start + step
        rows = coo.row[start:end]
        cols = coo.col[start:end]
        vs = coo.data[start:end]
        for r, c, v in zip(rows, cols, vs):
            triplets.append((ind2row[r], ind2col[c], v))
        save_triplets(wfilename, triplets, is_append=True)
        print 'finish saving 10M %s triplets in %s, progress: %s/%s, cost %.2f seconds' % (path_str, wfilename, (i+1) * step, len(coo.data), time.time() - start_time)

def save_comm_res(path_str, filename, comm_res, ind2row, ind2col):
    triplets = []
    coo = comm_res.tocoo()
    for r, c, v in zip(coo.row, coo.col,coo.data):
        triplets.append((ind2row[r], ind2col[c], v))
    save_triplets(filename, triplets)

def load_eids(eid_filename, type_):
    lines = open(eid_filename, 'r').readlines()
    eids = [int(l.strip()) for l in lines]
    eid2ind = {v:k for k,v in enumerate(eids)}
    ind2eid = reverse_map(eid2ind)
    #logger.info('get %s %s from %s', len(eids), type_, eid_filename)
    print 'get %s %s from %s' %(len(eids), type_, eid_filename)
    return eids, eid2ind, ind2eid

def get_bo(path_str, bid2ind):

    #U-pos-B-Cat-B
    if 'State' in path_str:
        sfilename = dir_ + 'bid_state.txt'
    elif 'Cat' in path_str:
        sfilename = dir_ + 'bid_cat.txt'
    elif 'City' in path_str:
        sfilename = dir_ + 'bid_city.txt'
    elif 'Star' in path_str:
        sfilename = dir_ + 'bid_stars.txt'

    lines = open(sfilename, 'r').readlines()
    parts = [l.strip().split() for l in lines]
    bos = [(int(b), int(o)) for b,o in parts]
    ond2ind = {v:k for k,v in enumerate(set([o for _, o in bos]))}
    ind2ond = reverse_map(ond2ind)
    adj_bo, adj_bo_t = generate_adj_mat(bos, bid2ind, ond2ind)
    return adj_bo, adj_bo_t

def cal_comm_mat_UBB(path_str):
    '''
        200k ratings
        calculate the commuting matrix in U-B-*-B style
        in fact, only need to calculate BB
    '''
    uid_filename = dir_ + 'uids.txt'
    print 'run cal_comm_mat_samples for 10k users in ', uid_filename
    lines = open(uid_filename, 'r').readlines()
    uids = [int(l.strip()) for l in lines]
    uid2ind = {v:k for k,v in enumerate(uids)}
    ind2uid = reverse_map(uid2ind)

    bid_filename = dir_ + 'bids.txt'
    lines = open(bid_filename, 'r').readlines()
    bids = [int(l.strip()) for l in lines]
    bid2ind = {v:k for k,v in enumerate(bids)}
    ind2bid = reverse_map(bid2ind)

    upb_filename = dir_ + 'uid_pos_bid.txt'
    upb = np.loadtxt(upb_filename, dtype=int)
    adj_ub, adj_ub_t = generate_adj_mat(upb, uid2ind, bid2ind)

    adj_bo, adj_bo_t = get_bo(path_str, bid2ind)

    t1 = time.time()
    comm_res = cal_mat_ubb(path_str, adj_ub, adj_bo, adj_bo_t)

    t2 = time.time()
    print 'cal res of %s cost %2.f seconds' % (path_str, t2 - t1)
    print 'comm_res shape=%s,densit=%s' % (comm_res.shape, comm_res.nnz * 1.0/comm_res.shape[0]/comm_res.shape[1])
    K = 500
    wfilename = dir_ + 'sim_res/path_count/%s_top%s.res' % (path_str, K)
    triplets = get_topK_items(comm_res, ind2uid, ind2bid, topK=K)
    save_triplets(wfilename, triplets)
    #batch_save_comm_res(path_str, wfilename, comm_res, ind2uid, ind2bid)
    t3 = time.time()
    print 'save res of %s cost %2.f seconds' % (path_str, t3 - t2)

def cal_comm_mat_UUB(path_str, cikm=False):
    '''
        calculate commuting matrix for U-*-U-pos-B style
    '''
    uid_filename = dir_ + 'uids.txt'
    bid_filename = dir_ + 'bids.txt'
    upb_filename = dir_ + 'uid_pos_bid.txt'
    if not cikm:
        rid_filename = dir_ + 'rids.txt'
        aid_filename = dir_ + 'aids.txt'

    print 'cal commut mat for %s, filenames: %s, %s, %s' % (path_str, uid_filename, bid_filename, upb_filename)
    uids, uid2ind, ind2uid = load_eids(uid_filename, 'user')
    bids, bid2ind, ind2bid = load_eids(bid_filename, 'biz')
    if not cikm:
        rids, rid2ind, ind2rid = load_eids(rid_filename, 'review')
        aids, aid2ind, ind2aid = load_eids(aid_filename, 'aspect')

    upb = np.loadtxt(upb_filename, dtype=np.int64)
    adj_upb, adj_upb_t = generate_adj_mat(upb, uid2ind, bid2ind)

    if path_str == 'UPBUB':
        start = time.time()
        UBU = adj_upb.dot(adj_upb_t)
        print 'UBU(%s), density=%.5f cost %.2f seconds' % (UBU.shape, UBU.nnz * 1.0/UBU.shape[0]/UBU.shape[1], time.time() - start)

    elif path_str in ['UPBCatBUB', 'UPBCityBUB']:
        start = time.time()
        adj_bo, adj_bo_t = get_bo(path_str, bid2ind)
        UBO = adj_upb.dot(adj_bo)
        UBU = UBO.dot(UBO.transpose())
        print 'UBU(%s), density=%.5f cost %.2f seconds' % (UBU.shape, UBU.nnz * 1.0/UBU.shape[0]/UBU.shape[1], time.time() - start)

    elif path_str in ['UNBCatBUB', 'UNBCityBUB']:
        unb_filename = dir_ + 'uid_neg_bid.txt'
        unb = np.loadtxt(unb_filename, dtype=np.int64)
        adj_unb, adj_unb_t = generate_adj_mat(unb, uid2ind, bid2ind)

        start = time.time()
        adj_bo, adj_bo_t = get_bo(path_str, bid2ind)
        UBO = adj_unb.dot(adj_bo)
        UBU = UBO.dot(UBO.transpose())
        print 'UBU(%s), density=%.5f cost %.2f seconds' % (UBU.shape, UBU.nnz * 1.0/UBU.shape[0]/UBU.shape[1], time.time() - start)

    elif path_str == 'UNBUB':
        unb_filename = dir_ + 'uid_neg_bid.txt'
        unb = np.loadtxt(unb_filename, dtype=np.int64)
        adj_unb, adj_unb_t = generate_adj_mat(unb, uid2ind, bid2ind)

        start = time.time()
        UBU = adj_unb.dot(adj_unb_t)
        print 'UBU(%s), density=%.5f cost %.2f seconds' % (UBU.shape, UBU.nnz * 1.0/UBU.shape[0]/UBU.shape[1], time.time() - start)

    elif path_str == 'UUB':
        social_filename = dir_ + 'user_social.txt'
        uu = np.loadtxt(social_filename, dtype=np.int64)
        adj_uu, adj_uu_t = generate_adj_mat(uu, uid2ind, uid2ind)

        start = time.time()
        UBU = adj_uu.dot(adj_uu_t)
        print 'UBU(%s), density=%.5f cost %.2f seconds' % (UBU.shape, UBU.nnz * 1.0/UBU.shape[0]/UBU.shape[1], time.time() - start)

    elif path_str == 'UCompUB':
        uid_comp_filename = dir_ + 'uid_comp.txt'
        uc = np.loadtxt(uid_comp_filename, dtype=np.int64)
        cids = set(uc[:,1])
        cid2ind = {v:k for k,v in enumerate(cids)}
        ind2cnd = reverse_map(cid2ind)
        adj_uc, adj_uc_t = generate_adj_mat(uc, uid2ind, cid2ind)

        start = time.time()
        UBU = adj_uc.dot(adj_uc_t)
        print 'UBU(%s), density=%.5f cost %.2f seconds' % (UBU.shape, UBU.nnz * 1.0/UBU.shape[0]/UBU.shape[1], time.time() - start)

    elif path_str == 'URPARUB':
        urpa_filename = dir_ + 'uid_rid_pos_aid.txt'
        urpa = np.loadtxt(urpa_filename)
        ur = list(set([(u,r) for u, r in urpa[:,(0,1)]]))# u, r multiple aspects, thus u-r can be duplicate
        adj_ur, adj_ur_t = generate_adj_mat(ur, uid2ind, rid2ind)
        ra = urpa[:,(1,2)]
        adj_ra, adj_ua_t = generate_adj_mat(ra, rid2ind, aid2ind)

        start = time.time()
        URA = adj_ur.dot(adj_ra)
        UBU = URA.dot(URA.transpose())#it should be URARU, here we use UBU for convenience
        print 'UBU(%s), density=%.5f cost %.2f seconds' % (UBU.shape, UBU.nnz * 1.0/UBU.shape[0]/UBU.shape[1], time.time() - start)

    elif path_str == 'URNARUB':
        urpa_filename = dir_ + 'uid_rid_neg_aid.txt'
        urpa = np.loadtxt(urpa_filename)
        ur = list(set([(u,r) for u, r in urpa[:,(0,1)]]))# u, r multiple aspects, thus u-r can be duplicate
        adj_ur, adj_ur_t = generate_adj_mat(ur, uid2ind, rid2ind)
        ra = urpa[:,(1,2)]
        adj_ra, adj_ua_t = generate_adj_mat(ra, rid2ind, aid2ind)

        start = time.time()
        URA = adj_ur.dot(adj_ra)
        UBU = URA.dot(URA.transpose())#it should be URARU, here we use UBU for convenience
        print 'UBU(%s), density=%.5f cost %.2f seconds' % (UBU.shape, UBU.nnz * 1.0/UBU.shape[0]/UBU.shape[1], time.time() - start)

    start = time.time()
    UBUB = UBU.dot(adj_upb)
    print 'UBUB(%s), density=%.5f cost %.2f seconds' % (UBUB.shape, UBUB.nnz * 1.0/UBUB.shape[0]/UBUB.shape[1], time.time() - start)
    start = time.time()
    K = 500
    triplets = get_topK_items(UBUB, ind2uid, ind2bid, topK=K)
    wfilename = dir_ + 'sim_res/path_count/%s_top%s.res' % (path_str, K)
    save_triplets(wfilename, triplets)
    #save_comm_res(path_str, wfilename, UBUB, ind2uid, ind2bid)
    print 'finish saving %s %s entries in %s, cost %.2f seconds' % (len(triplets), path_str, wfilename, time.time() - start)

def cal_comm_mat_USUB(path_str):
    '''
        Given meta_structure_str, generate the commuting matrix
        e.g. 'user-review-business,t10_aspect-review-user'
    '''

    uid_filename = dir_ + 'uids.txt'
    bid_filename = dir_ + 'bids.txt'
    aid_filename = dir_ + 'aids.txt'
    rid_filename = dir_ + 'rids.txt'
    upb_filename = dir_ + 'uid_pos_bid.txt'

    print 'cal commut mat for %s, filenames: %s, %s, %s' % (path_str, uid_filename, bid_filename, upb_filename)
    uids, uid2ind, ind2uid = load_eids(uid_filename, 'user')
    bids, bid2ind, ind2bid = load_eids(bid_filename, 'biz')
    aids, aid2ind, ind2aid = load_eids(aid_filename, 'aspect')

    upb = np.loadtxt(upb_filename, dtype=np.int64)
    adj_upb, adj_upb_t = generate_adj_mat(upb, uid2ind, bid2ind)

    if 'P' in path_str:
        urb_filename = dir_ + 'uid_rid_pos_bid.txt'
        ura_filename = dir_ + 'uid_rid_pos_aid.txt'
        ind2rid_filename = dir_ + 'sim_res/path_count/%s_spa_mat_id_map.pickle' % path_str
        rar_mat_filename = dir_ + 'sim_res/path_count/%s_spa_mat.pickle' % path_str
    elif 'N' in path_str:
        urb_filename = dir_ + 'uid_rid_neg_bid.txt'
        ura_filename = dir_ + 'uid_rid_neg_aid.txt'
        ind2rid_filename = dir_ + 'sim_res/path_count/%s_spa_mat_id_map.pickle' % path_str
        rar_mat_filename = dir_ + 'sim_res/path_count/%s_spa_mat.pickle' % path_str

    f = open(ind2rid_filename, 'r')
    ind2rid = pickle.load(f)
    rid2ind = reverse_map(ind2rid)

    urb = np.loadtxt(urb_filename, dtype=np.int64)
    ura = np.loadtxt(ura_filename, dtype=np.int64)

    ur = urb[:,(0,1)]
    adj_ur, adj_ur_t = generate_adj_mat(ur, uid2ind, rid2ind)

    rb = urb[:,(1,2)]
    adj_rb, adj_rb_t = generate_adj_mat(rb, rid2ind, bid2ind)

    ra = ura[:,(1,2)]
    adj_ra, adj_ra_t = generate_adj_mat(ra, rid2ind, aid2ind)

    start = time.time()
    RBR = adj_rb.dot(adj_rb_t)
    print 'RBR(%s), density=%.5f cost %.2f seconds' % (RBR.shape, RBR.nnz * 1.0/RBR.shape[0]/RBR.shape[1], time.time() - start)
    start = time.time()
    #RAR = adj_ra.dot(adj_ra_t)
    f = open(rar_mat_filename, 'r')
    RAR = pickle.load(f)
    print 'load RAR(%s), density=%.5f cost %.2f seconds' % (RAR.shape, RAR.nnz * 1.0/RAR.shape[0]/RAR.shape[1], time.time() - start)
    start = time.time()
    RSR = RBR.multiply(RAR)
    print 'RSR(%s), density=%.5f cost %.2f seconds' % (RSR.shape, RSR.nnz * 1.0/RSR.shape[0]/RSR.shape[1], time.time() - start)
    start = time.time()
    URSR = adj_ur.dot(RSR)
    print 'URSR(%s), density=%.5f cost %.2f seconds' % (URSR.shape, URSR.nnz * 1.0/URSR.shape[0]/URSR.shape[1], time.time() - start)
    start = time.time()
    URSRU = URSR.dot(adj_ur_t)
    print 'URSRU(%s), density=%.5f cost %.2f seconds' % (URSRU.shape, URSRU.nnz * 1.0/URSRU.shape[0]/URSRU.shape[1], time.time() - start)

    start = time.time()
    URSRUB = URSRU.dot(adj_upb)
    print 'URSRUB(%s), density=%.5f cost %.2f seconds' % (URSRUB.shape, URSRUB.nnz * 1.0/URSRUB.shape[0]/URSRUB.shape[1], time.time() - start)
    start = time.time()
    K = 500
    wfilename = dir_ + 'sim_res/path_count/%s_top%s.res' % (path_str, K)
    #wfilename = dir_ + 'sim_res/path_count/%s.res' % path_str
    batch_save_comm_res(path_str, wfilename, URSRUB, ind2uid, ind2bid)
    print 'finish saving %s %s entries in %s, cost %.2f seconds' % (URSRUB.nnz, path_str, wfilename, time.time() - start)

def cal_rar(path_str):

    aid_filename = dir_ + 'aids.txt'
    rid_filename = dir_ + 'rids.txt'

    aids = open(aid_filename, 'r').readlines()
    aids = [int(r.strip()) for r in aids]
    aid2ind = {a:ind for ind, a in enumerate(aids)}#global ind
    ind2aid = reverse_map(aid2ind)

    rids = open(rid_filename, 'r').readlines()
    rids = [int(r.strip()) for r in rids]
    rid2ind = {r:ind for ind, r in enumerate(rids)}#global ind
    ind2rid = reverse_map(rid2ind)

    if 'P' in path_str:
        ura_filename = dir_ + 'uid_rid_pos_aid_weight.txt'
    elif 'N' in path_str:
        ura_filename = dir_ + 'uid_rid_neg_aid_weight.txt'

    ura = np.loadtxt(ura_filename, dtype=np.float64)
    ra = ura[:,(1,2,3)]
    ra = [(int(r), int(a), w) for r, a, w in ra]
    adj_ra, adj_ra_t = generate_adj_mat(ra, rid2ind, aid2ind, is_weight=True)

    t1 = time.time()
    RA = adj_ra.toarray()
    t2 = time.time()
    print 'to dense RA%s cost %.2f seconds' % (RA.shape, t2 - t1)
    RAR_csr = cal_rar_block(RA, len(rid2ind), ind2rid, step=20000)
    print 'finish cal rar by blocks, cost %.2f minutes' % ((time.time() - t2) / 60.0)
    try:
        wfilename = dir_ + 'sim_res/path_count/%s_spa_mat.pickle' % path_str
        fw = open(wfilename, 'w+')
        pickle.dump(RAR_csr, fw, pickle.HIGHEST_PROTOCOL)
        map_filename = dir_ + 'sim_res/path_count/%s_spa_mat_id_map.pickle' % path_str
        fw = open(map_filename, 'w+')
        pickle.dump(ind2rid, fw, pickle.HIGHEST_PROTOCOL)
        print 'finish saving sparse mat in ', wfilename
    except Exception as e:
        print e

DEBUG = False
def cal_rar_block(RA, nR, ind2rid, step=10000, topK=100):
    if DEBUG:
        RA = np.random.rand(1005,10)
        ind2rid = {k:k for k in range(1005)}
        nR = 1005
        step, topK = 20, 10
        debug_RR = np.dot(RA, RA.T)
        col_inds = bn.argpartsort(-debug_RR, topK, axis=1)[:,:topK]
        dr,dc = col_inds.shape
        row_inds = np.tile(np.arange(dr).reshape(dr,1), dc)
        debug_res = np.zeros((1005,1005))
        debug_res[row_inds, col_inds] = 1

    step_num = RA.shape[0] / step
    data, rows, cols = [],[],[]
    rar_start = time.time()
    for i in range(step_num+1):
        r = i * step
        rblock = RA[r:r+step]

        b_top100_res = []
        b_top100_inds = []
        tmp_res = {}
        #finish 10000 users
        block_start = time.time()
        for j in range(step_num+1):
            c = j * step
            cblock = RA[c:c+step]
            t3 = time.time()

            dot_res = np.dot(rblock, cblock.T)# dot res: 10000 * 10000

            drc = dot_res.shape[1]
            tmp_topK = topK if topK < drc else drc

            top100_inds = bn.argpartsort(-dot_res, tmp_topK, axis=1)[:,:tmp_topK]#10000 * 100,100 indices of the top K weights, column indices in dot_res
            br, bc = top100_inds.shape
            top100_rows = np.tile(np.arange(br).reshape(br,1), bc)#number of colums = colums of top100 inds, usually =100

            top100_res = dot_res[top100_rows, top100_inds]#only need to preserve top 100 weights for global comparing

            b_top100_res.append(top100_res)

            b_top100_inds.append(top100_inds + c)#preserve the global indices, indices need to add the starting value of every block

        block_end = time.time()
        print 'finish calculating %s-th/%s block(%s*%s), cost %.2f seconds, rar_block cost %.2f minutes' % (i+1, step_num, step, step, block_end - block_start, (block_end - rar_start) / 60.0)
        b_top100_inds = np.concatenate(b_top100_inds, axis=1)
        b_top100_res = np.concatenate(b_top100_res, axis=1)

        top100_inds = bn.argpartsort(-b_top100_res, topK, axis=1)[:,:topK]#10000 * 100,100 indices of the top K weights
        tr, tc = top100_inds.shape
        #it may exists that not all 100 weights are zero, prob is very small, processing later
        top100_rows = np.tile(np.arange(tr).reshape(tr,1), tc)

        #global row and col inds are needed for the constructing the sparse matrix for RAR 
        top100_res = b_top100_res[top100_rows, top100_inds]#10000 * 100, some may equal zero
        b_col_top100_inds = b_top100_inds[top100_rows, top100_inds]#global column inds for top100, then we need to get global row inds

        #the following code is used for gurantee all the weights > 0.0, remove 0 weights, very uncommon
        trows, tcols = np.where(top100_res > 0.0)#return all the rows and cols of top100_res
        global_col_inds = b_col_top100_inds[trows, tcols]#value corresponded to the trows + r
        global_row_inds = trows + i * step
        rows.extend(global_row_inds)
        cols.extend(global_col_inds)

        triplets = []
        save_start = time.time()
        print 'finish selecting top %s for block %s, cost %.2f seconds, rar_block cost %.2f minutes' % (topK, i+1, save_start - block_end, (save_start - rar_start) / 60.0)
        #for r, c in zip(global_row_inds, global_col_inds):
        #    triplets.append((ind2rid[r], ind2rid[c], 1))
        #filename = dir_ + 'sim_res/path_count/%s_block_res/%s.dat' % (i+1)
        #save_triplets(filename, triplets)
        #save_end = time.time()
        #print 'finish processing block %s, res saved in %s, %s triplets, cost detail(total/compute/select/save): %.2f/%.2f/%.2f/%.2f seconds ' % (i+1, filename, len(triplets), save_end - block_start, block_end - block_start, save_start - block_end, save_end - save_start)

    data = np.ones(len(rows))#use one to replace the weights

    t4 = time.time()
    RAR_csr = csr((data, (rows, cols)), shape=[nR, nR])
    t5 = time.time()
    #print '10000 res to sparse matrix(%s) cost %.2f seconds' % (RAR_csr.shape, t5 - t4)
    if DEBUG:
        test = RAR_csr.toarray()
        test_res = (test == debug_res)
        if test_res.sum() == test.size:
            print '!!!block matrix equals directly dot matrix, the res is correct!!!'
        else:
            print 'two matrices are not equal, the res is wrong!!!'
    # make it symmetryic
    RAR = 0.5 * (RAR_csr + RAR_csr.transpose())
    RAR = RAR.ceil()#0.5 to 1, 1 to 1
    if DEBUG:
        import pdb;pdb.set_trace()
    return RAR

def cal_yelp_all(split_num, dt):
    global dir_
    dir_ = 'data/%s/exp_split/%s/' % (dt, split_num)

    for path_str in ['UPBCatB','UPBCityB', 'UPBStateB', 'UPBStarsB']:
        cal_comm_mat_UBB(path_str)

    for path_str in ['UPBUB', 'UNBUB', 'URPARUB', 'URNARUB', 'UUB']:
        cal_comm_mat_UUB(path_str)

    for path_str in ['URPSRUB', 'URNSRUB']:
        cal_rar(path_str)
        cal_comm_mat_USUB(path_str)

if __name__ == '__main__':
    if len(sys.argv) == 4:
        dt = sys.argv[1]
        path_str = sys.argv[2]
        split_num = int(sys.argv[3])
        if dt == 'UBB':
            cal_comm_mat_UBB(path_str)
        elif dt == 'UUB':
            cal_comm_mat_UUB(path_str)
        elif dt == 'USUB':
            cal_comm_mat_USUB(path_str)
        elif dt == 'RAR':
            cal_rar(path_str)
        elif path_str == 'all':
            cal_yelp_all(split_num, dt)
