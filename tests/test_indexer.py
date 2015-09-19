#!/usr/bin/env python
# coding: utf-8

#########################################################################
#########################################################################

"""
   File Name: test_indexer.py
      Author: Wan Ji
      E-mail: wanji@live.com
  Created on: Tue Nov  4 10:38:24 2014 CST
"""
DESCRIPTION = """
"""

import os
import shutil
import unittest

import time
import logging

import numpy as np
import scipy.io as spio
# import cProfile

from hdidx.distance import distFunc
from hdidx import indexer


def load_random(ntrain, nbase, nquery, d=16):
    """
    Generate a set of unit norm vectors
    """
    np.random.seed(0)
    vtrain = np.random.random((ntrain, d))
    vbase = np.random.random((nbase, d))
    vquery = np.random.random((nquery, d))

    t0 = time.clock()
    ids_gnd = np.empty(nquery)
    logging.info("Computing the ground-truth...")
    batsize = 20
    for q in range(0, nquery, batsize):
        logging.info("\r\t%d/%d" % (q, nquery))
        last = min(q+batsize, nquery)
        dist = distFunc['euclidean'](vbase, vquery[q:last])
        ids_gnd[q:last] = dist.argmin(1)
    logging.info("\r\t%d/%d\tDone!" % (nquery, nquery))
    # dis_gnd = [dist[i, ids_gnd[i]] for i in range(dist.shape[0])]
    tgnd = time.clock() - t0
    logging.info("GND Time: %.3fs" % tgnd)
    return vtrain, vbase, vquery, ids_gnd


def create_random_data(ntrain=10**4, nbase=10**4, nquery=10**2):
    """
    Create random data
    """
    # synthetic dataset
    vtrain, vbase, vquery, ids_gnd = load_random(ntrain, nbase, nquery)
    spio.savemat('./test-tmp/hdidx_test_vbase.mat', {'feat': vbase[:10, :]})

    return np.require(vtrain, np.single, requirements="C"),\
        np.require(vbase, np.single, requirements="C"),    \
        np.require(vquery, np.single, requirements="C"),   \
        ids_gnd


def compute_stats(nquery, ids_gnd, ids_pqc, k):
    nn_ranks_pqc = np.zeros(nquery)
    qry_range = np.arange(ids_pqc.shape[1])
    for i in range(nquery):
        gnd_ids = ids_gnd[i]
        nn_pos = qry_range[ids_pqc[i, :] == gnd_ids]

        if len(nn_pos) == 1:
            nn_ranks_pqc[i] = nn_pos
        else:
            nn_ranks_pqc[i] = k + 1

    nn_ranks_pqc.sort()

    for i in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]:
        if i > k:
            break
        r_at_i = (nn_ranks_pqc < i).sum() * 100.0 / nquery
        logging.warning('\rr@%-3d = %.3f' % (i, r_at_i))


class TestPQNew(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    @classmethod
    def setUpClass(cls):
        cls.vtrain, cls.vbase, cls.vquery, cls.ids_gnd = \
            create_random_data()
        cls.nsubq = 8
        cls.topk = 100
        cls.coarsek = 32

    @classmethod
    def tearDownClass(cls):
        pass

    def test_ivfpq_lmdb_0_build_save_add_search(self):
        """ Test IVFPQ: LMDB storage
                from scratch
        """

        # create indexer
        idx = indexer.IVFPQIndexer()
        # building code books
        idx.build({
            'vals': self.vtrain,
            'nsubq': self.nsubq,
            'coarsek': self.coarsek,
        })
        # saving indexer to disk file
        idx.save('./test-tmp/hdidx_test_ivf_lmdb.info')
        # set backend storage
        idx.set_storage('lmdb', {
            'path': './test-tmp/hdidx_test_ivf_lmdb.idx',
            'clear': True,
        })
        # indexing
        idx.add(self.vbase)
        # search
        ids, dis = idx.search(self.vquery, topk=self.topk)

        # cProfile.runctx('ids, dis = idx.search(self.vquery, topk=self.topk)',
        #                 None, locals())

        # evaluate
        compute_stats(self.vquery.shape[0], self.ids_gnd, ids, self.topk)

    def test_ivfpq_lmdb_1_SKIP_load_add_search(self):
        """ Test IVFPQ: LMDB storage
                load pre-computed quantizers from disk file
        """
        # create indexer
        idx = indexer.IVFPQIndexer()
        # load indexer from disk file
        idx.load('./test-tmp/hdidx_test_ivf_lmdb.info')
        # set backend storage
        idx.set_storage('lmdb', {
            'path': './test-tmp/hdidx_test_ivf_lmdb.idx',
            'clear': True,
        })
        # indexing
        idx.add(self.vbase)
        # search
        ids, dis = idx.search(self.vquery, topk=self.topk)
        # evaluate
        compute_stats(self.vquery.shape[0], self.ids_gnd, ids, self.topk)

    def test_ivfpq_lmdb_2_SKIP_load_SKIP_search(self):
        """ Test IVFPQ: LMDB storage
                1. load pre-computed quantizers from disk file
                2. load indices from LMDB
        """
        # create indexer
        idx = indexer.IVFPQIndexer()
        # load indexer from disk file
        idx.load('./test-tmp/hdidx_test_ivf_lmdb.info')
        # set backend storage
        idx.set_storage('lmdb', {
            'path': './test-tmp/hdidx_test_ivf_lmdb.idx',
            'clear': False,
        })
        # search
        ids, dis = idx.search(self.vquery, topk=self.topk)
        # evaluate
        compute_stats(self.vquery.shape[0], self.ids_gnd, ids, self.topk)

    def test_ivfpq_mem(self):
        """ Test IVFPQ: memory storage
        """
        idx = indexer.IVFPQIndexer()

        idx.build({
            'vals': self.vtrain,
            'nsubq': self.nsubq,
            'coarsek': self.coarsek,
        })
        idx.save('./test-tmp/hdidx_test_ivf_mem.info')
        idx.set_storage('mem', {})

        idx.add(self.vbase)
        ids, dis = idx.search(self.vquery, topk=self.topk)
        compute_stats(self.vquery.shape[0], self.ids_gnd, ids, self.topk)

    def test_pq_lmdb_0_build_save_add_search(self):
        """ Test PQ: LMDB storage
                from scratch
        """
        idx = indexer.PQIndexer()
        idx.build({
            'vals': self.vtrain,
            'nsubq': self.nsubq,
        })
        idx.save('./test-tmp/hdidx_test_lmdb.info')
        idx.set_storage('lmdb', {
            'path': './test-tmp/hdidx_test_lmdb.idx',
            'clear': True,
        })
        idx.add(self.vbase)
        ids, dis = idx.search(self.vquery, topk=self.topk)
        compute_stats(self.vquery.shape[0], self.ids_gnd, ids, self.topk)

    def test_pq_lmdb_1_SKIP_load_add_search(self):
        """ Test PQ: LMDB storage
                load pre-computed quantizers from disk file
        """
        idx = indexer.PQIndexer()
        idx.load('./test-tmp/hdidx_test_lmdb.info')
        idx.set_storage('lmdb', {
            'path': './test-tmp/hdidx_test_lmdb.idx',
            'clear': True,
        })
        idx.add(self.vbase)
        ids, dis = idx.search(self.vquery, topk=self.topk)
        compute_stats(self.vquery.shape[0], self.ids_gnd, ids, self.topk)

    def test_pq_lmdb_2_SKIP_load_SKIP_search(self):
        """ Test PQ: LMDB storage
                1. load pre-computed quantizers from disk file
                2. load indices from LMDB
        """
        idx = indexer.PQIndexer()
        idx.load('./test-tmp/hdidx_test_lmdb.info')
        idx.set_storage('lmdb', {
            'path': './test-tmp/hdidx_test_lmdb.idx',
            'clear': False,
        })
        ids, dis = idx.search(self.vquery, topk=self.topk)
        compute_stats(self.vquery.shape[0], self.ids_gnd, ids, self.topk)

    def test_pq_mem(self):
        """ Test PQ: memory storage
                from scratch
        """
        idx = indexer.PQIndexer()
        idx.build({
            'vals': self.vtrain,
            'nsubq': self.nsubq,
        })
        idx.save('./test-tmp/hdidx_test_mem.info')
        idx.set_storage('mem')

        idx.add(self.vbase)
        ids, dis = idx.search(self.vquery, topk=self.topk)
        compute_stats(self.vquery.shape[0], self.ids_gnd, ids, self.topk)


class TestSH(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    @classmethod
    def setUpClass(cls):
        cls.vtrain, cls.vbase, cls.vquery, cls.ids_gnd = \
            create_random_data()
        cls.nbits = 64
        cls.topk = 100

    @classmethod
    def tearDownClass(cls):
        pass

    def test_sh_lmdb_0_build_save_add_search(self):
        """ Test PQ: LMDB storage
                from scratch
        """
        idx = indexer.SHIndexer()
        idx.build({
            'vals': self.vtrain,
            'nbits': self.nbits,
        })
        idx.save('./test-tmp/hdidx_test_sh_lmdb.info')
        idx.set_storage('lmdb', {
            'path': './test-tmp/hdidx_test_sh_lmdb.idx',
            'clear': True,
        })
        idx.add(self.vbase)
        ids, dis = idx.search(self.vquery, topk=self.topk)
        compute_stats(self.vquery.shape[0], self.ids_gnd, ids, self.topk)

    def test_sh_lmdb_1_SKIP_load_add_search(self):
        """ Test PQ: LMDB storage
                load pre-computed quantizers from disk file
        """
        idx = indexer.SHIndexer()
        idx.load('./test-tmp/hdidx_test_sh_lmdb.info')
        idx.set_storage('lmdb', {
            'path': './test-tmp/hdidx_test_sh_lmdb.idx',
            'clear': True,
        })
        idx.add(self.vbase)
        ids, dis = idx.search(self.vquery, topk=self.topk)
        compute_stats(self.vquery.shape[0], self.ids_gnd, ids, self.topk)

    def test_sh_lmdb_2_SKIP_load_SKIP_search(self):
        """ Test PQ: LMDB storage
                1. load pre-computed quantizers from disk file
                2. load indices from LMDB
        """
        idx = indexer.SHIndexer()
        idx.load('./test-tmp/hdidx_test_sh_lmdb.info')
        idx.set_storage('lmdb', {
            'path': './test-tmp/hdidx_test_sh_lmdb.idx',
            'clear': False,
        })
        ids, dis = idx.search(self.vquery, topk=self.topk)
        compute_stats(self.vquery.shape[0], self.ids_gnd, ids, self.topk)

    def test_sh_mem(self):
        """ Test PQ: memory storage
                from scratch
        """
        idx = indexer.SHIndexer()
        idx.build({
            'vals': self.vtrain,
            'nbits': self.nbits,
        })
        idx.save('./test-tmp/hdidx_test_mem.info')
        idx.set_storage('mem')

        idx.add(self.vbase)
        ids, dis = idx.search(self.vquery, topk=self.topk)
        compute_stats(self.vquery.shape[0], self.ids_gnd, ids, self.topk)


if __name__ == '__main__':
    logging.warn("The results of mem storage and lmdb storage might be " +
                 "different even if the database and queries are exactly " +
                 "the same, this is because the randomization exists in " +
                 "k-means clustering.")
    TEMPDIR = "./test-tmp"
    if os.path.exists(TEMPDIR):
        shutil.rmtree(TEMPDIR)
    os.makedirs(TEMPDIR)
    unittest.main(failfast=True)
    # cProfile.run('unittest.main(failfast=True)')
