#coding=utf8
'''
    utils
'''

import numpy as np
from scipy.sparse import csr_matrix as csr

def reverse_map(m):
    return {v:k for k,v in m.items()}

def generate_adj_mat(relation, row_map, col_map, is_weight=False):
    data, rows, cols = [],[],[]
    for r in relation:
        if is_weight:
            data.append(r[2])
        else:
            data.append(1)
        rows.append(row_map[r[0]])
        cols.append(col_map[r[1]])
    adj = csr((data,(rows,cols)),shape=[len(row_map), len(col_map)])
    adj_t = csr((data,(cols, rows)),shape=[len(col_map), len(row_map)])
    return adj, adj_t

def load_rand_data():
    '''
        return the features, labels, and the group inds
    '''
    S, N = 1000, 80
    X = np.random.normal(size=[S,N])
    Y = np.random.uniform(size=[S])

    test_X = np.random.normal(size=[200, N])
    test_Y = np.random.uniform(size=[200])
    logger.info('train_data: (%.4f,%.4f), test_data: (%.4f,%.4f)', np.mean(Y), np.std(Y), np.mean(test_Y), np.std(test_Y))
    return X, Y, test_X, test_Y

def save_lines(filename, res):
    fw = open(filename, 'w+')
    fw.write('\n'.join(res))
    fw.close()
    print 'save %s lines in %s' % (len(res), filename)

def save_triplets(filename, triplets, is_append=False):
    if is_append:
        fw = open(filename, 'a+')
        fw.write('\n')
    else:
        fw = open(filename, 'w+')
    fw.write('\n'.join(['%s\t%s\t%s' % (h,t,v) for h,t,v in triplets]))
    fw.close()
    print 'save %s triplets in %s' % (len(triplets), filename)

def test_save_triplets():
    a = [(i,i**2, i**3) for i in range(10)]
    filename = 'log/test_appending_mode2.txt'
    for ind in xrange(0, len(a), 3):
        tri = a[ind:ind+3]
        save_triplets(filename, tri, is_append=True)

if __name__ == '__main__':
    test_save_triplets()

