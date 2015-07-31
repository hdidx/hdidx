#!/usr/bin/env python
# coding: utf-8

"""
   File Name: sh.py
      Author: Wan Ji
      E-mail: wanji@live.com
  Created on: Mon Jul 27 10:22:06 2015 CST
"""
DESCRIPTION = """
"""

import logging

import numpy as np

from hdidx.indexer import Indexer
from hdidx.util import Profiler, eigs
from hdidx.storage import createStorage

import hdidx._cext as cext

BIT_CNT_MAP = np.array([bin(i).count("1") for i in xrange(256)], np.uint16)


class SHIndexer(Indexer):
    def __init__(self):
        Indexer.__init__(self)
        self.set_storage()

    def __del__(self):
        pass

    def build(self, pardic=None):
        # training data
        X = pardic['vals']
        # the number of subquantizers
        nbits = pardic['nbits']
        # the number of items in one block
        blksize = pardic.get('blksize', 16384)

        [Nsamples, Ndim] = X.shape

        # algo:
        # 1) PCA
        npca = min(nbits, Ndim)
        pc, l = eigs(np.cov(X.T), npca)
        X = X.dot(pc)   # no need to remove the mean

        # 2) fit uniform distribution
        eps = np.finfo(float).eps
        mn = np.percentile(X, 5)
        mx = np.percentile(X, 95)
        mn = X.min(0) - eps
        mx = X.max(0) + eps

        # 3) enumerate eigenfunctions
        R = mx - mn
        maxMode = np.ceil((nbits+1) * R / R.max())
        nModes = maxMode.sum() - maxMode.size + 1
        modes = np.ones((nModes, npca))
        m = 0
        for i in xrange(npca):
            modes[m+1:m+maxMode[i], i] = np.arange(1, maxMode[i]) + 1
            m = m + maxMode[i] - 1
        modes = modes - 1
        omega0 = np.pi / R
        omegas = modes * omega0.reshape(1, -1).repeat(nModes, 0)
        eigVal = -(omegas ** 2).sum(1)
        ii = (-eigVal).argsort()
        modes = modes[ii[1:nbits+1], :]

        """
        Initializing indexer data
        """
        idxdat = dict()
        idxdat['nbits'] = nbits
        idxdat['pc'] = pc
        idxdat['mn'] = mn
        idxdat['mx'] = mx
        idxdat['modes'] = modes
        idxdat['blksize'] = blksize
        self.idxdat = idxdat

    def set_storage(self, storage_type='mem', storage_parm=None):
        self.storage = createStorage(storage_type, storage_parm)

    @staticmethod
    def compactbit(b):
        nSamples, nbits = b.shape
        nwords = (nbits + 7) / 8
        B = np.hstack([np.packbits(b[:, i*8:(i+1)*8][:, ::-1], 1)
                       for i in xrange(nwords)])
        residue = nbits % 8
        if residue != 0:
            B[:, -1] = np.right_shift(B[:, -1], 8 - residue)
            print 8 - residue

        return B

    def compressSH(self, vals):
        X = vals
        if X.ndim == 1:
            X = X.reshape((1, -1))

        Nsamples, Ndim = X.shape
        nbits = self.idxdat['nbits']
        mn = self.idxdat['mn']
        mx = self.idxdat['mx']
        pc = self.idxdat['pc']
        modes = self.idxdat['modes']

        X = X.dot(pc)
        X = X - mn.reshape((1, -1))
        omega0 = 0.5 / (mx - mn)
        omegas = modes * omega0.reshape((1, -1))

        U = np.zeros((Nsamples, nbits))
        for i in range(nbits):
            omegai = omegas[i, :]
            ys = X * omegai + 0.25
            ys -= np.floor(ys)
            yi = np.sum(ys < 0.5, 1)
            U[:, i] = yi

        b = np.require(U % 2 == 0, dtype=np.int)
        B = self.compactbit(b)
        return B

    def add(self, vals, keys=None):
        num_vals = vals.shape[0]
        if keys is None:
            num_base_items = self.storage.get_num_items()
            keys = np.arange(num_base_items, num_base_items + num_vals,
                             dtype=np.int32)
        else:
            keys = np.array(keys, dtype=np.int32).reshape(-1)

        blksize = self.idxdat.get('blksize', 16384)
        start_id = 0
        for start_id in range(0, num_vals, blksize):
            cur_num = min(blksize, num_vals - start_id)
            logging.info("%8d/%d: %d" % (start_id, num_vals, cur_num))
            codes = self.compressSH(vals[start_id:start_id+cur_num, :])
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

    def search(self, queries, topk=None, thresh=None):
        nq = queries.shape[0]
        nbits = self.idxdat['nbits']

        # qry_codes = self.compressSH(queries)
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
            qry_code = self.compressSH(queries[qry_id:qry_id+1])
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
                    '\t%d/%d: %.4fs per query' %
                    (qry_id+1, nq, profiler.sum_average()))
                logging.info("\t\t%s" % profiler.str_average())
                profiler.reset()
        logging.info('Querying Finished!')
        time_total += profiler.sum_overall()
        logging.info("Average querying time: %.4f" % (time_total / nq))

        return ids, dis
