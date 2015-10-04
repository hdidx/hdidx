#!/usr/bin/env python
# coding: utf-8

"""
   File Name: hamming.py
      Author: Wan Ji
      E-mail: wanji@live.com
  Created on: Mon Jul 27 10:22:06 2015 CST
"""
DESCRIPTION = """
Indexers for binary codes in Hamming space.
"""

import os
import logging
import cPickle as pickle
import operator

import numpy as np

import hdidx.encoder
from hdidx.indexer import Indexer
from hdidx.util import Profiler
from hdidx.storage import createStorage

import hdidx._cext as cext
from hdidx import _mih as mih

BIT_CNT_MAP = np.array([bin(i).count("1") for i in xrange(256)], np.uint16)
DEFAULT_HAMMING_ENCODER = "SHEncoder"


class SHIndexer(Indexer):
    def __init__(self, encoder=DEFAULT_HAMMING_ENCODER):
        Indexer.__init__(self)
        self.encoder = getattr(hdidx.encoder, encoder)()
        self.set_storage()

    def __del__(self):
        pass

    def build(self, pardic=None):
        self.encoder.build(pardic)

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

        start_id = 0
        for start_id in range(0, num_vals, self.BLKSIZE):
            cur_num = min(self.BLKSIZE, num_vals - start_id)
            logging.info("%8d/%d: %d" % (start_id, num_vals, cur_num))
            codes = self.encoder.encode(vals[start_id:start_id+cur_num, :])
            self.storage.add(codes, keys[start_id:start_id+cur_num])

    def remove(self, keys):
        raise Exception(self.ERR_UNIMPL)

    @staticmethod
    def hammingDist(B1, B2):
        """
        Compute hamming distance between two sets of samples (B1, B2)

        Dh=hammingDist(B1, B2);

        Input
        B1, B2: compact bit vectors. Each datapoint is one row.
        size(B1) = [ndatapoints1, nwords]
        size(B2) = [ndatapoints2, nwords]
        It is faster if ndatapoints1 < ndatapoints2

        Output
        Dh = hamming distance.
        size(Dh) = [ndatapoints1, ndatapoints2]

        example query
        Dhamm = hammingDist(B2, B1);
        this will give the same result than:
        Dhamm = distMat(U2>0, U1>0).^2;
        the size of the distance matrix is:
        size(Dhamm) = [Ntest x Ntraining]
        """

        if B1.ndim == 1:
            B1 = B1.reshape((1, -1))

        if B2.ndim == 1:
            B2 = B2.reshape((1, -1))

        npt1, dim1 = B1.shape
        npt2, dim2 = B2.shape

        if dim1 != dim2:
            raise Exception("Dimension not consists: %d, %d" % (dim1, dim2))

        Dh = np.zeros((npt1, npt2), np.uint16)

        for i in xrange(npt1):
            Dh[i, :] = BIT_CNT_MAP[np.bitwise_xor(B1[i, :], B2)].sum(1)

        return Dh

    @staticmethod
    def hammingDist2(B1, B2):
        """
        Compute hamming distance between two sets of samples (B1, B2)

        Dh=hammingDist(B1, B2);

        Input
        B1, B2: compact bit vectors. Each datapoint is one row.
        size(B1) = [ndatapoints1, nwords]
        size(B2) = [ndatapoints2, nwords]
        It is faster if ndatapoints1 < ndatapoints2

        Output
        Dh = hamming distance.
        size(Dh) = [ndatapoints1, ndatapoints2]

        example query
        Dhamm = hammingDist(B2, B1);
        this will give the same result than:
        Dhamm = distMat(U2>0, U1>0).^2;
        the size of the distance matrix is:
        size(Dhamm) = [Ntest x Ntraining]
        """

        if B1.ndim == 1:
            B1 = B1.reshape((1, -1))

        if B2.ndim == 1:
            B2 = B2.reshape((1, -1))

        npt1, dim1 = B1.shape
        npt2, dim2 = B2.shape

        if dim1 != dim2:
            raise Exception("Dimension not consists: %d, %d" % (dim1, dim2))

        Dh = cext.hamming(B1, B2)

        return Dh

    def search(self, queries, topk=None, **kwargs):
        nq = queries.shape[0]
        nbits = self.encoder.ecdat['nbits']

        # qry_codes = self.encoder.encode(queries)
        db_codes = self.storage.get_codes()
        idsquerybase = self.storage.get_keys()

        dis = np.ones((nq, topk), np.single) * np.inf
        ids = np.ones((nq, topk), np.int32) * -1

        profiler = Profiler()
        interval = 100 if nq >= 100 else 10
        time_total = 0.0    # total time for all queries
        logging.info('Start Querying ...')
        for qry_id in range(nq):
            profiler.start("encoding")  # time for computing the distances
            qry_code = self.encoder.encode(queries[qry_id:qry_id+1])
            profiler.end()

            profiler.start("distance")  # time for computing the distances
            disquerybase = self.hammingDist2(qry_code, db_codes).reshape(-1)
            profiler.end()

            profiler.start("knn")       # time for finding the kNN
            cur_ids = cext.knn_count(disquerybase, nbits, topk)
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


class MIHIndexer(Indexer):
    def __init__(self, encoder=DEFAULT_HAMMING_ENCODER):
        Indexer.__init__(self)
        self.encoder = getattr(hdidx.encoder, encoder)()
        self.key_map = mih.get_key_map(16)
        self.set_storage()

    def __del__(self):
        pass

    def build(self, pardic=None):
        self.encoder.build(pardic)

    def set_storage(self, storage_type='mem', storage_parm=None):
        self.storage = createStorage(storage_type, storage_parm)
        if storage_parm is None:
            return
        self.idx_path = storage_parm['path'] + ".mih"
        if os.path.exists(self.idx_path):
            with open(self.idx_path, 'rb') as idxf:
                self.tables = pickle.load(idxf)
            self.ntbls = len(self.tables)
        else:
            self.ntbls = self.encoder.ecdat['nbits'] / 16
            self.tables = [dict() for i in xrange(self.ntbls)]
            with open(self.idx_path, 'wb') as idxf:
                pickle.dump(self.tables, idxf)

    def add(self, vals, keys=None):
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
            codes = self.encoder.encode(vals[start_id:start_id+cur_num, :])
            self.storage.add(codes, keys[start_id:start_id+cur_num])
            codes_16bit = mih.get_keys_16bit(codes)
            for i in xrange(cur_num):
                for j in xrange(self.ntbls):
                    subcode = codes_16bit[i, j]
                    if subcode in self.tables[j]:
                        self.tables[j][subcode].append(start_id+i)
                    else:
                        self.tables[j][subcode] = [start_id+i]

        with open(self.idx_path, 'wb') as idxf:
            pickle.dump(self.tables, idxf)

    def remove(self, keys):
        raise Exception(self.ERR_UNIMPL)

    def search(self, queries, topk=None, **kwargs):
        nq = queries.shape[0]
        nbits = self.encoder.ecdat['nbits']
        # qry_codes = self.encoder.encode(queries)
        db_codes = self.storage.get_codes()
        idsquerybase = self.storage.get_keys()
        idmap = idsquerybase.argsort()

        dis = np.ones((nq, topk), np.single) * np.inf
        ids = np.ones((nq, topk), np.int32) * -1

        profiler = Profiler()
        interval = 100 if nq >= 100 else 10
        time_total = 0.0    # total time for all queries
        logging.info('Start Querying ...')
        for qry_id in range(nq):
            profiler.start("encoding")    # time for getting final result
            qry_code = self.encoder.encode(queries[qry_id:qry_id+1])
            qry_keys = mih.get_keys_16bit(qry_code)

            last_subth = -1
            ret_set = [[] for i in xrange(nbits+1)]
            proced = set()
            acc = 0
            profiler.end()

            for hmdist in xrange(nbits+1):
                profiler.start("pre")    # time for
                subth = hmdist / self.ntbls
                if subth <= last_subth:
                    continue
                profiler.end()

                profiler.start("reduce")    # time for
                # sel_keys = reduce(operator.concat, [
                #     [
                #         mask ^ subcode
                #         for mask in self.key_map[subth]
                #     ] for table, subcode in zip(self.tables, qry_keys[0])
                # ])
                candidates = reduce(operator.concat, [
                    reduce(operator.concat, [
                        table.get(mask ^ subcode, [])
                        for mask in self.key_map[subth]
                    ]) for table, subcode in zip(self.tables, qry_keys[0])
                ])
                profiler.end()

                profiler.start("filter")    # time for
                filtered = []
                for cur_id in candidates:
                    if cur_id not in proced:
                        filtered.append(cur_id)
                        proced.add(cur_id)

                candidates = db_codes[idmap[filtered]]
                profiler.end()

                profiler.start("calc")    # time for
                cur_dist = list(cext.hamming(qry_code, candidates).reshape(-1))
                # print subth, sorted(zip(cur_dist, filtered),
                #                     key=lambda x: x[0])[:10]
                for i, d in zip(filtered, cur_dist):
                    ret_set[d].append(i)
                acc += len(ret_set[hmdist])
                if acc >= topk:
                    break
                last_subth = subth
                profiler.end()

            profiler.start("result")    # time for getting final result
            shift = 0
            for hmdist in xrange(nbits+1):
                cur_num = min(topk - shift, len(ret_set[hmdist]))
                dis[qry_id, shift:shift+cur_num] = hmdist
                ids[qry_id, shift:shift+cur_num] = ret_set[hmdist][:cur_num]
                shift += cur_num
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
