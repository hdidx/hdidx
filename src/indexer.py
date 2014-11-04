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

# import ipdb
import cPickle
import logging

import numpy as np

from util import kmeans, pq_kmeans_assign, pq_knn
from distance import distFunc


class Indexer(object):
    class IdxData(object):
        pass

    def __init__(self):
        self.INST_ERR = "Instance of `Indexer` is not allowed!"
        pass

    def __del__(self):
        pass

    def build(self, vals=None, labels=None):
        """
        Build the indexer based on given training data
        """
        raise Exception(self.INST_ERR)

    def load(self, path):
        """
        Load indexer information from file
        """
        with open(path, 'rb') as pklf:
            self.idxdat = cPickle.load(pklf)

    def save(self, path):
        """
        Saving the information related to the indexer itself
        """
        with open(path, 'wb') as pklf:
            cPickle.dump(self.idxdat, pklf)

    def add(self, vals, keys):
        """
        Add one or more items to the indexer
        """
        raise Exception(self.INST_ERR)

    def remove(self, keys):
        """
        Remove one or more items from the indexer
        """
        raise Exception(self.INST_ERR)

    def search(self, query, topk=None, thresh=None):
        """
        Search in the indexer for `k` nearest neighbors or
        neighbors in a distance of `thresh`
        """
        raise Exception(self.INST_ERR)


class SampleIndexer(Indexer):
    def __init__(self):
        self.INST_ERR = "Instance of `Indexer` is not allowed!"
        pass

    def __del__(self):
        pass

    def build(self, vals=None, labels=None):
        """
        Create the indexer based on given training data
        """
        pass

    def add(self, vals, keys):
        """
        Add one or more items to the indexer
        """
        pass

    def remove(self, keys):
        """
        Remove one or more items from the indexer
        """
        pass

    def search(self, query, topk=None, thresh=None):
        """
        Search in the indexer for `k` nearest neighbors or
        neighbors in a distance of `thresh`
        """
        pass


class PQIndexer(Indexer):
    def __init__(self):
        self.INST_ERR = "Instance of `Indexer` is not allowed!"
        pass

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
        idxdat = Indexer.IdxData()
        idxdat.nsubq = nsubq
        idxdat.ksub = ksub
        idxdat.dsub = dsub
        idxdat.centroids = [None for q in range(nsubq)]
        idxdat.blksize = blksize

        logging.info("Building codebooks in subspaces - BEGIN\n")
        for q in range(nsubq):
            logging.info("\tsubspace %d/%d:\n" % (q, nsubq))
            vs = np.require(vals[:, q*dsub:(q+1)*dsub], requirements='C')
            idxdat.centroids[q] = kmeans(vs, ksub, niter=100)

        logging.info("Building codebooks in subspaces - DONE\n")

        self.idxdat = idxdat
        self.blocks = []
        self.keys = []
        self.num_emptys = 0

    def add(self, vals, keys=None):
        num_vals = vals.shape[0]
        start_id = 0

        blksize = self.idxdat.blksize
        dsub = self.idxdat.dsub
        nsubq = self.idxdat.nsubq
        centroids = self.idxdat.centroids

        while start_id < num_vals:
            if self.num_emptys is 0:
                self.blocks.append(np.zeros((blksize, nsubq), np.uint8))
                self.num_emptys = blksize
            cur_num = min(self.num_emptys, num_vals-start_id)
            cur_start = blksize - self.num_emptys

            for q in range(nsubq):
                vsub = vals[start_id:start_id+cur_num, q*dsub:(q+1)*dsub]
                # ipdb.set_trace()
                self.blocks[-1][cur_start:cur_start+cur_num, q] = \
                    pq_kmeans_assign(centroids[q], vsub)

            self.num_emptys -= cur_num
            start_id += cur_num
        # ipdb.set_trace()

    def remove(self, keys):
        raise Exception(self.INST_ERR)

    def search(self, query, topk=None, thresh=None):
        nq = query.shape[0]

        dsub = self.idxdat.dsub
        nsubq = self.idxdat.nsubq
        ksub = self.idxdat.ksub
        centroids = self.idxdat.centroids

        distab = np.zeros((nsubq, ksub), np.single)
        dis = np.zeros((nq, topk), np.single)
        ids = np.zeros((nq, topk), np.single)

        for qid in range(nq):
            # pre-compute the table of squared distance to centroids
            for q in range(nsubq):
                vsub = query[qid:qid+1, q*dsub:(q+1)*dsub]
                distab[q:q+1, :] = distFunc['euclidean'](
                    centroids[q], vsub)

            # add the tabulated distances to construct the distance estimators
            disquerybase = self.sumidxtab(distab)
            cur_ids = pq_knn(disquerybase, topk)

            dis[qid, :] = disquerybase[cur_ids]
            ids[qid, :] = cur_ids

        return ids, dis

    def sumidxtab(self, D):
        """
        Compute distance to database items based on distances to centroids.
            D: nsubq x ksub
        """

        nsubq = D.shape[0]
        blksize = self.idxdat.blksize
        num_base_items = len(self.blocks) * blksize - self.num_emptys

        dis = np.zeros(num_base_items)

        for ib, blk in zip(range(0, num_base_items, blksize), self.blocks):
            cur_num = min(blksize, num_base_items-ib)
            dis[ib:ib+cur_num] = [sum([D[j, blk[i, j]] for j in range(nsubq)])
                                  for i in range(cur_num)]

        return dis
