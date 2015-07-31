#!/usr/bin/env python
# coding: utf-8

"""
   File Name: pq.py
      Author: Wan Ji
      E-mail: wanji@live.com
  Created on: Fri Jul 31 20:26:32 2015 CST
"""
DESCRIPTION = """
"""

import logging

import numpy as np

from hdidx.encoder import Encoder
from hdidx.util import kmeans, pq_kmeans_assign


class PQEncoder(Encoder):
    def __init__(self):
        Encoder.__init__(self)

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
        ecdat = dict()
        ecdat['nsubq'] = nsubq
        ecdat['ksub'] = ksub
        ecdat['dsub'] = dsub
        ecdat['blksize'] = blksize
        ecdat['centroids'] = [None for q in range(nsubq)]

        logging.info("Building codebooks in subspaces - BEGIN")
        for q in range(nsubq):
            logging.info("\tsubspace %d/%d:" % (q, nsubq))
            vs = np.require(vals[:, q*dsub:(q+1)*dsub],
                            requirements='C', dtype=np.float32)
            ecdat['centroids'][q] = kmeans(vs, ksub, niter=100)
        logging.info("Building codebooks in subspaces - DONE")

        self.ecdat = ecdat

    def encode(self, vals):
        dsub = self.ecdat['dsub']
        nsubq = self.ecdat['nsubq']
        centroids = self.ecdat['centroids']

        num_vals = vals.shape[0]
        codes = np.zeros((num_vals, nsubq), np.uint8)
        for q in range(nsubq):
            vsub = vals[:, q*dsub:(q+1)*dsub]
            codes[:, q] = pq_kmeans_assign(centroids[q], vsub)
        return codes
