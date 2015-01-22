#!/usr/bin/env python
# coding: utf-8

#########################################################################
#########################################################################

"""
   File Name: indexer.py
      Author: Wan Ji
      E-mail: wanji@live.com
  Created on: Tue Nov  4 09:07:38 2014 CST
"""
DESCRIPTION = """
"""

import cPickle as pickle
import logging
import time

import numpy as np

from util import kmeans, pq_kmeans_assign, pq_knn
from distance import distFunc
from storage import createStorage

import _cext as cext


class Indexer(object):
    class IdxData(object):
        pass

    def __init__(self):
        self.ERR_INSTAN = "Instance of `Indexer` is not allowed!"
        self.ERR_UNIMPL = "Unimplemented method!"
        pass

    def __del__(self):
        pass

    def build(self, vals=None, labels=None):
        """
        Build the indexer based on given training data
        """
        raise Exception(self.ERR_INSTAN)

    def load(self, path):
        """
        Load indexer information from file
        """
        with open(path, 'rb') as pklf:
            self.idxdat = pickle.load(pklf)

    def save(self, path):
        """
        Save the information related to the indexer itself
        """
        with open(path, 'wb') as pklf:
            pickle.dump(self.idxdat, pklf, protocol=2)

    def set_storage(self, storage_type='mem', storage_parm=None):
        """
        Set up the backend storage engine
        """
        raise Exception(self.ERR_INSTAN)

    def add(self, vals, keys):
        """
        Add one or more items to the indexer
        """
        raise Exception(self.ERR_INSTAN)

    def remove(self, keys):
        """
        Remove one or more items from the indexer
        """
        raise Exception(self.ERR_INSTAN)

    def search(self, queries, topk=None, thresh=None):
        """
        Search in the indexer for `k` nearest neighbors or
        neighbors in a distance of `thresh`
        """
        raise Exception(self.ERR_INSTAN)


class PQIndexer(Indexer):
    def __init__(self):
        Indexer.__init__(self)

    def __del__(self):
        pass

    def build(self, pardic=None):
        # training data
        vals = pardic['vals']
        # the number of subquantizers
        nsubq = pardic['nsubq']
        # the number bits of each subquantizer
        nsubqbits = pardic.get('nsubqbits', 8)
        # the number of items in one block

        blksize = pardic.get('blksize', 16384)

        # vector dimension
        dim = vals.shape[1]
        # dimension of the subvectors to quantize
        dsub = dim / nsubq
        # number of centroids per subquantizer
        ksub = 2 ** nsubqbits

        """
        Initializing indexer data
        """
        idxdat = dict()
        idxdat['nsubq'] = nsubq
        idxdat['ksub'] = ksub
        idxdat['dsub'] = dsub
        idxdat['blksize'] = blksize
        idxdat['centroids'] = [None for q in range(nsubq)]

        logging.info("Building codebooks in subspaces - BEGIN")
        for q in range(nsubq):
            logging.info("\tsubspace %d/%d:" % (q, nsubq))
            vs = np.require(vals[:, q*dsub:(q+1)*dsub], requirements='C')
            idxdat['centroids'][q] = kmeans(vs, ksub, niter=100)
        logging.info("Building codebooks in subspaces - DONE")

        self.idxdat = idxdat

    def set_storage(self, storage_type='mem', storage_parm=None):
        self.storage = createStorage(storage_type, storage_parm)

    def add(self, vals, keys=None):
        num_vals = vals.shape[0]
        if keys is None:
            num_base_items = self.storage.get_num_items()
            keys = np.arange(num_base_items, num_base_items + num_vals,
                             dtype=np.int32)
        else:
            keys = np.array(keys, dtype=np.int32).reshape(-1)

        dsub = self.idxdat['dsub']
        nsubq = self.idxdat['nsubq']
        centroids = self.idxdat['centroids']

        blksize = self.idxdat.get('blksize', 16384)
        start_id = 0
        for start_id in range(0, num_vals, blksize):
            cur_num = min(blksize, num_vals - start_id)
            print "%8d/%d: %d" % (start_id, num_vals, cur_num)

            codes = np.zeros((cur_num, nsubq), np.uint8)
            for q in range(nsubq):
                vsub = vals[start_id:start_id+cur_num, q*dsub:(q+1)*dsub]
                codes[:, q] = pq_kmeans_assign(centroids[q], vsub)
            self.storage.add(codes, keys[start_id:start_id+cur_num])

    def remove(self, keys):
        raise Exception(self.ERR_UNIMPL)

    def search(self, queries, topk=None, thresh=None):
        nq = queries.shape[0]

        dsub = self.idxdat['dsub']
        nsubq = self.idxdat['nsubq']
        ksub = self.idxdat['ksub']
        centroids = self.idxdat['centroids']

        distab = np.zeros((nsubq, ksub), np.single)
        dis = np.ones((nq, topk), np.single) * np.inf
        ids = np.ones((nq, topk), np.int32) * -1

        interval = 100 if nq >= 100 else 10
        dbg_qnt = []
        dbg_sit = []
        dbg_knn = []
        dbg_ret = []
        logging.info('Start Querying ...')
        time_start = time.time()
        for qry_id in range(nq):
            t0 = time.time()
            # pre-compute the table of squared distance to centroids
            for qnt_id in range(nsubq):
                vsub = queries[qry_id:qry_id+1, qnt_id*dsub:(qnt_id+1)*dsub]
                distab[qnt_id:qnt_id+1, :] = distFunc['euclidean'](
                    centroids[qnt_id], vsub)
            dbg_qnt.append(time.time() - t0)

            t0 = time.time()
            # add the tabulated distances to construct the distance estimators
            idsquerybase, disquerybase = self.sumidxtab(distab)
            dbg_sit.append(time.time() - t0)

            t0 = time.time()
            cur_ids = pq_knn(disquerybase, topk)
            dbg_knn.append(time.time() - t0)

            t0 = time.time()
            ids[qry_id, :] = idsquerybase[cur_ids]
            dis[qry_id, :] = disquerybase[cur_ids]
            dbg_ret.append(time.time() - t0)

            if (qry_id+1) % interval == 0:
                logging.info(
                    '\t%d/%d: %.4fs per query' %
                    (qry_id+1, nq, (time.time() - time_start) / interval))
                time_start = time.time()
                logging.debug("\tquantizing:%.4f" % np.mean(dbg_qnt))
                logging.debug("\tsumidxtab: %.4f" % np.mean(dbg_sit))
                logging.debug("\t   pq_knn: %.4f" % np.mean(dbg_knn))
                logging.debug("\t   genret: %.4f" % np.mean(dbg_ret))
                dbg_qnt = []
                dbg_sit = []
                dbg_knn = []
                dbg_ret = []
        logging.info('Querying Finished!')

        return ids, dis

    def sumidxtab(self, D):
        """
        Compute distance to database items based on distances to centroids.
            D: nsubq x ksub
        """

        ids = self.storage.get_keys()
        dis = cext.sumidxtab_core(D, self.storage.get_codes())

        return np.array(ids), np.array(dis)

        """
        Deprecated code
        """
        # num_base_items = self.storage.get_num_items()

        # dis = np.zeros(num_base_items)
        # ids = np.arange(0)

        # start_id = 0
        # for keys, blk in self.storage:
        #     cur_num = blk.shape[0]
        #     # dis[start_id:start_id+cur_num] = self.sumidxtab_core(D, blk)
        #     dis[start_id:start_id+cur_num] = cext.sumidxtab_core(D, blk)
        #     start_id += cur_num
        #     ids = np.hstack((ids, keys))

        # return ids, dis

    @classmethod
    def sumidxtab_core(cls, D, blk):
        # return 0
        return [sum([D[j, blk[i, j]] for j in range(D.shape[0])])
                for i in range(blk.shape[0])]


class IVFPQIndexer(PQIndexer):
    def __init__(self):
        PQIndexer.__init__(self)

    def __del__(self):
        pass

    def build(self, pardic=None):
        # training data
        vals = pardic['vals']
        # the number of coarse centroids
        coarsek = pardic['coarsek']

        logging.info('Building coarse quantizer - BEGIN')
        coa_centroids = kmeans(vals, coarsek, niter=100)
        cids = pq_kmeans_assign(coa_centroids, vals)
        logging.info('Building coarse quantizer - DONE')

        pardic['vals'] -= coa_centroids[cids, :]

        PQIndexer.build(self, pardic)

        self.idxdat['coa_centroids'] = coa_centroids
        self.idxdat['coarsek'] = coarsek

    def set_storage(self, storage_type='mem', storage_parm=None):
        storage_parm['coarsek'] = self.idxdat['coarsek']
        self.storage = createStorage(storage_type, storage_parm)

    def add(self, vals, keys=None):
        num_vals = vals.shape[0]
        if keys is None:
            num_base_items = sum([ivf.get_num_items() for ivf in self.storage])
            keys = np.arange(num_base_items, num_base_items + num_vals,
                             dtype=np.int32)
        else:
            keys = np.array(keys, dtype=np.int32).reshape(-1)

        dsub = self.idxdat['dsub']
        nsubq = self.idxdat['nsubq']
        centroids = self.idxdat['centroids']
        coarsek = self.idxdat['coarsek']
        coa_centroids = self.idxdat['coa_centroids']

        blksize = self.idxdat.get('blksize', 16384)
        start_id = 0
        for start_id in range(0, num_vals, blksize):
            cur_num = min(blksize, num_vals - start_id)
            end_id = start_id + cur_num
            print "%8d/%d: %d" % (start_id, num_vals, cur_num)

            # Here `copy()` can ensure that you DONOT modify the vals
            cur_vals = vals[start_id:end_id, :].copy()
            cur_cids = pq_kmeans_assign(coa_centroids, cur_vals)
            cur_vals -= coa_centroids[cur_cids, :]

            codes = np.zeros((cur_num, nsubq), np.uint8)
            for q in range(nsubq):
                vsub = cur_vals[:, q*dsub:(q+1)*dsub]
                codes[:, q] = pq_kmeans_assign(centroids[q], vsub)
            # self.storage.add(codes, keys)
            for ivfidx in xrange(coarsek):
                self.storage[ivfidx].add(
                    codes[cur_cids == ivfidx, :],
                    keys[start_id:end_id][cur_cids == ivfidx])

    def remove(self, keys):
        raise Exception(self.ERR_UNIMPL)

    def search(self, queries, topk=None, thresh=None, nn_coa=8):
        nq = queries.shape[0]

        dsub = self.idxdat['dsub']
        nsubq = self.idxdat['nsubq']
        ksub = self.idxdat['ksub']
        centroids = self.idxdat['centroids']
        coa_centroids = self.idxdat['coa_centroids']

        distab = np.zeros((nsubq, ksub), np.single)
        dis = np.ones((nq, topk), np.single) * np.inf
        ids = np.ones((nq, topk), np.int32) * -1

        coa_dist = distFunc['euclidean'](coa_centroids, queries)
        logging.info('Start Querying ...')
        time_start = time.time()
        interval = 100 if nq >= 100 else 10
        for qry_id in range(nq):
            # Here `copy()` can ensure that you DONOT modify the queries
            query = queries[qry_id:qry_id+1, :].copy()
            coa_knn = pq_knn(coa_dist[qry_id, :], nn_coa)
            query = query - coa_centroids[coa_knn, :]
            v_idsquerybase = []
            v_disquerybase = []
            for coa_idx in range(nn_coa):
                # pre-compute the table of squared distance to centroids
                for qnt_id in range(nsubq):
                    vsub = query[coa_idx:coa_idx+1,
                                 qnt_id*dsub:(qnt_id+1)*dsub]
                    distab[qnt_id:qnt_id+1, :] = distFunc['euclidean'](
                        centroids[qnt_id], vsub)

                # construct the distance estimators from tabulated distances
                idsquerybase, disquerybase = self.sumidxtab(
                    distab, coa_knn[coa_idx])
                v_idsquerybase.append(idsquerybase)
                v_disquerybase.append(disquerybase)

            idsquerybase = np.hstack(tuple(v_idsquerybase))
            disquerybase = np.hstack(tuple(v_disquerybase))
            realk = min(disquerybase.shape[0], topk)
            cur_ids = pq_knn(disquerybase, realk)

            ids[qry_id, :realk] = idsquerybase[cur_ids]
            dis[qry_id, :realk] = disquerybase[cur_ids]
            if (qry_id+1) % interval == 0:
                logging.info(
                    '\t%d/%d: %.4fs per query' %
                    (qry_id+1, nq, (time.time() - time_start) / interval))
                time_start = time.time()
        logging.info('Querying Finished!')

        return ids, dis

    def sumidxtab(self, D, ivfidx):
        """
        Compute distance to database items based on distances to centroids.
            D: nsubq x ksub
        """

        ids = self.storage[ivfidx].get_keys()
        if ids.shape[0] == 0:
            dis = np.ndarray(0)
        else:
            dis = cext.sumidxtab_core(D, self.storage[ivfidx].get_codes())

        return np.array(ids), np.array(dis)

        """
        Deprecated code
        """
        num_candidates = self.storage[ivfidx].get_num_items()
        dis = np.zeros(num_candidates)
        ids = np.arange(0)

        start_id = 0

        for keys, blk in self.storage[ivfidx]:
            cur_num = blk.shape[0]
            # dis[start_id:start_id+cur_num] = self.sumidxtab_core(D, blk)
            dis[start_id:start_id+cur_num] = cext.sumidxtab_core(D, blk)
            start_id += cur_num
            ids = np.hstack((ids, keys))

        return ids, dis
