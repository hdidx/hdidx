#!/usr/bin/env python
# coding: utf-8

"""
   File Name: vq.py
      Author: Wan Ji
      E-mail: wanji@live.com
  Created on: Mon Jul 27 10:22:00 2015 CST
"""
DESCRIPTION = """
Indexers for Vector Quantization algorithms.
"""

import logging

import numpy as np

from hdidx.indexer import Indexer
from hdidx.encoder import PQEncoder, IVFPQEncoder
from hdidx.util import pq_knn, Profiler
from hdidx.distance import distFunc
# from hdidx.distance import fast_euclidean
from hdidx._cext import fast_euclidean
from hdidx.storage import createStorage

import hdidx._cext as cext


class PQIndexer(Indexer):
    def __init__(self):
        Indexer.__init__(self)
        self.encoder = PQEncoder()
        self.storage = None

    def __del__(self):
        pass

    def build(self, pardic=None):
        self.encoder.build(pardic)

    def set_storage(self, storage_type='mem', storage_parm=None):
        self.storage = createStorage(storage_type, storage_parm)

    def add(self, vals, keys=None):
        if self.storage is None:
            self.set_storage()

        num_vals = vals.shape[0]
        if keys is None:
            num_base_items = self.storage.get_num_items()
            keys = np.arange(num_base_items, num_base_items + num_vals,
                             dtype=np.int32)
        else:
            keys = np.array(keys, dtype=np.int32).reshape(-1)

        start_id = 0
        for start_id in range(0, num_vals, self.BLKSIZE):
            cur_num = min(self.BLKSIZE, num_vals - start_id)
            logging.info("%8d/%d: %d" % (start_id, num_vals, cur_num))

            codes = self.encoder.encode(vals[start_id:start_id+cur_num])
            self.storage.add(codes, keys[start_id:start_id+cur_num])

    def remove(self, keys):
        raise Exception(self.ERR_UNIMPL)

    def search(self, queries, topk=None,  **kwargs):
        nq = queries.shape[0]

        dsub = self.encoder.ecdat['dsub']
        nsubq = self.encoder.ecdat['nsubq']
        ksub = self.encoder.ecdat['ksub']
        centroids = self.encoder.ecdat['centroids']

        distab = np.zeros((nsubq, ksub), np.single)
        dis = np.ones((nq, topk), np.single) * np.inf
        ids = np.ones((nq, topk), np.int32) * -1

        profiler = Profiler()
        interval = 100 if nq >= 100 else 10
        time_total = 0.0    # total time for all queries
        logging.info('Start Querying ...')
        for qry_id in xrange(nq):
            profiler.start("distab")    # time for computing distance table
            # pre-compute the table of squared distance to centroids
            for qnt_id in range(nsubq):
                vsub = queries[qry_id:qry_id+1, qnt_id*dsub:(qnt_id+1)*dsub]
                distab[qnt_id:qnt_id+1, :] = distFunc['euclidean'](
                    centroids[qnt_id], vsub)
            profiler.end()

            profiler.start("distance")  # time for computing the distances
            # add the tabulated distances to construct the distance estimators
            idsquerybase, disquerybase = self.sumidxtab(distab)
            profiler.end()

            profiler.start("knn")       # time for finding the kNN
            realk = min(disquerybase.shape[0], topk)
            cur_ids = pq_knn(disquerybase, realk)
            profiler.end()

            profiler.start("result")    # time for getting final result
            ids[qry_id, :] = idsquerybase[cur_ids]
            dis[qry_id, :] = disquerybase[cur_ids]
            profiler.end()

            if (qry_id+1) % interval == 0:
                time_total += profiler.sum_overall()
                logging.info(
                    '\t%d/%d: %.3fms per query' %
                    (qry_id+1, nq, profiler.sum_average() * 1000))
                logging.info("\t\t%s" % profiler.str_average())
                profiler.reset()
        logging.info('Querying Finished!')
        time_total += profiler.sum_overall()
        logging.info("Average querying time: %.3fms" % (time_total * 1000 / nq))

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
        self.encoder = IVFPQEncoder()
        self.storage = None

    def __del__(self):
        pass

    def build(self, pardic=None):
        self.encoder.build(pardic)

    def set_storage(self, storage_type='mem', storage_parm=None):
        if storage_parm is None:
            storage_parm = dict()
        storage_parm['num_idx'] = self.encoder.ecdat['coarsek']
        self.storage = createStorage(storage_type, storage_parm)

    def add(self, vals, keys=None):
        if self.storage is None:
            self.set_storage()

        num_vals = vals.shape[0]
        if keys is None:
            num_base_items = sum([ivf.get_num_items() for ivf in self.storage])
            keys = np.arange(num_base_items, num_base_items + num_vals,
                             dtype=np.int32)
        else:
            keys = np.array(keys, dtype=np.int32).reshape(-1)

        start_id = 0
        logging.info("Building indexes - BEGIN")
        for start_id in range(0, num_vals, self.BLKSIZE):
            cur_num = min(self.BLKSIZE, num_vals - start_id)
            end_id = start_id + cur_num
            logging.info("%8d/%d: %d" % (start_id, num_vals, cur_num))

            cids, codes = self.encoder.encode(vals[start_id:end_id, :])

            coarsek = self.encoder.ecdat['coarsek']
            for ivfidx in xrange(coarsek):
                self.storage[ivfidx].add(
                    codes[cids == ivfidx, :],
                    keys[start_id:end_id][cids == ivfidx])
        logging.info("Building indexes - DONE!")

    def remove(self, keys):
        raise Exception(self.ERR_UNIMPL)

    def search(self, queries, topk=None, **kwargs):
        nn_coa = kwargs.get('nn_coa', 8)
        nq = queries.shape[0]

        dsub = self.encoder.ecdat['dsub']
        nsubq = self.encoder.ecdat['nsubq']
        ksub = self.encoder.ecdat['ksub']
        centroids = self.encoder.ecdat['centroids']
        coa_centroids = self.encoder.ecdat['coa_centroids']

        centroids_l2norm = []
        for i in xrange(nsubq):
            centroids_l2norm.append((centroids[i] ** 2).sum(1))
        coa_centroids_l2norm = (coa_centroids ** 2).sum(1)

        distab = np.zeros((nsubq, ksub), np.single)
        dis = np.ones((nq, topk), np.single) * np.inf
        ids = np.ones((nq, topk), np.int32) * -1

        profiler = Profiler()
        interval = 100 if nq >= 100 else 10
        time_total = 0.0    # total time for all queries
        logging.info('Start Querying ...')
        for qry_id in xrange(nq):
            # Here `copy()` can ensure that you DONOT modify the queries
            query = queries[qry_id:qry_id+1, :].copy()
            profiler.start("coa_knn")
            coa_dist = fast_euclidean(
                coa_centroids, query, coa_centroids_l2norm).reshape(-1)
            # profiler.end()
            # profiler.start("coa_knn_knn")
            coa_knn = pq_knn(coa_dist, nn_coa)
            profiler.end()

            profiler.start("distab+distance")
            query = query - coa_centroids[coa_knn, :]
            v_idsquerybase = []
            v_disquerybase = []
            for coa_idx in range(nn_coa):
                # pre-compute the table of squared distance to centroids
                for qnt_id in range(nsubq):
                    vsub = query[coa_idx:coa_idx+1,
                                 qnt_id*dsub:(qnt_id+1)*dsub]
                    distab[qnt_id:qnt_id+1, :] = fast_euclidean(
                        centroids[qnt_id], vsub, centroids_l2norm[qnt_id])

                # construct the distance estimators from tabulated distances
                idsquerybase, disquerybase = self.sumidxtab(
                    distab, coa_knn[coa_idx])
                v_idsquerybase.append(idsquerybase)
                v_disquerybase.append(disquerybase)

            idsquerybase = np.hstack(tuple(v_idsquerybase))
            disquerybase = np.hstack(tuple(v_disquerybase))
            profiler.end()

            profiler.start("knn")       # time for finding the kNN
            realk = min(disquerybase.shape[0], topk)
            cur_ids = pq_knn(disquerybase, realk)
            profiler.end()

            profiler.start("result")    # time for getting final result
            ids[qry_id, :realk] = idsquerybase[cur_ids]
            dis[qry_id, :realk] = disquerybase[cur_ids]
            profiler.end()

            if (qry_id+1) % interval == 0:
                time_total += profiler.sum_overall()
                logging.info(
                    '\t%d/%d: %.3fms per query' %
                    (qry_id+1, nq, profiler.sum_average() * 1000))
                logging.info("\t\t%s" % profiler.str_average())
                profiler.reset()
        logging.info('Querying Finished!')
        time_total += profiler.sum_overall()
        logging.info("Average querying time: %.3fms" % (time_total * 1000 / nq))

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
