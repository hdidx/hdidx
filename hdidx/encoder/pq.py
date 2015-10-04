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
    """
    Product Quantization Encoder
    """
    def __init__(self):
        super(PQEncoder, self).__init__()

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
            logging.info("\tsubspace %d/%d" % (q, nsubq))
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


class IVFPQEncoder(PQEncoder):
    """
    Inverted-File Product Quantization Encoder
    """
    def __init__(self):
        super(IVFPQEncoder, self).__init__()

    def __del__(self):
        pass

    def build(self, pardic=None):
        pardic['vals'] = pardic['vals'].copy()
        # training data
        vals = pardic['vals']
        # the number of coarse centroids
        coarsek = pardic.get('coarsek', 1024)

        logging.info('Building coarse quantizer - BEGIN')
        coa_centroids = kmeans(vals.astype(np.float32), coarsek, niter=100)
        cids = pq_kmeans_assign(coa_centroids, vals)
        logging.info('Building coarse quantizer - DONE')

        pardic['vals'] -= coa_centroids[cids, :]
        super(IVFPQEncoder, self).build(pardic)

        self.ecdat['coa_centroids'] = coa_centroids
        self.ecdat['coarsek'] = coarsek

    def encode(self, vals):
        # Here `copy()` can ensure that you DONOT modify the vals
        vals = vals.copy()

        coa_centroids = self.ecdat['coa_centroids']

        cids = pq_kmeans_assign(coa_centroids, vals)
        vals -= coa_centroids[cids, :]

        codes = super(IVFPQEncoder, self).encode(vals)
        return cids, codes
