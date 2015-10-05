#!/usr/bin/env python
# coding: utf-8

#########################################################################
#########################################################################

"""
   File Name: _cext.pyx
      Author: Wan Ji
      E-mail: wanji@live.com
  Created on: Wed Nov  5 16:32:11 2014 CST
"""
DESCRIPTION = """
"""

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as cnp

# use the Numpy-C-API from Cython
cnp.import_array()

# cdefine the signature of our c function
cdef extern from "cext.h":
    void sumidxtab_core_cfunc(const cnp.float32_t * D, const cnp.uint8_t * blk,
        int nsq, int ksub, int cur_num, cnp.float32_t * out)
    void hamming_core_cfunc(cnp.uint8_t * qry, cnp.uint8_t * db, int dim, int num,
                            cnp.uint16_t * dist)
    void knn_count_core_cfunc(cnp.uint16_t * D, int numD, int maxD,
                              int topk, cnp.int32_t * out);
    void fast_euclidean_core_cfunc(const cnp.float32_t * feat, const cnp.float32_t * query,
                                   const cnp.float32_t * featl2norm, int dim, int num,
                                   cnp.float32_t * dist);

# create the wrapper code, with numpy type annotations
def sumidxtab_core(cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] D not None,
                   cnp.ndarray[cnp.uint8_t, ndim=2, mode="c"] blk not None):
    out = np.zeros(blk.shape[0], dtype=D.dtype)
    sumidxtab_core_cfunc(<cnp.float32_t*> cnp.PyArray_DATA(D),
                         <cnp.uint8_t*> cnp.PyArray_DATA(blk),
                         int(D.shape[0]), int(D.shape[1]), int(blk.shape[0]),
                         <cnp.float32_t*> cnp.PyArray_DATA(out))
    return out


def hamming(cnp.ndarray[cnp.uint8_t, ndim=2, mode="c"] Q not None,
            cnp.ndarray[cnp.uint8_t, ndim=2, mode="c"] D not None):
    """
    Calculate Hamming Distance
    """
    nq, dq = Q.shape[0], Q.shape[1]
    nd, dd = D.shape[0], D.shape[1]
    out = np.zeros((nq, nd), dtype=np.uint16)
    dist = np.zeros(nd, dtype=np.uint16)
    for i in xrange(nq):
        hamming_core_cfunc(<cnp.uint8_t*> cnp.PyArray_DATA(Q[i, :]),
                           <cnp.uint8_t*> cnp.PyArray_DATA(D),
                           int(dq), int(nd),
                           <cnp.uint16_t*> cnp.PyArray_DATA(out[i, :]))
    return out

def knn_count(cnp.ndarray[cnp.uint16_t, ndim=1, mode="c"] D not None,
              int maxD, int topk):
    """
    Return topk by counting sort
    """
    out = np.zeros(topk, dtype=np.int32)
    knn_count_core_cfunc(<cnp.uint16_t*> cnp.PyArray_DATA(D),
                         int(D.shape[0]), maxD, topk,
                         <cnp.int32_t*> cnp.PyArray_DATA(out))
    return out


def fast_euclidean(feat, query, featl2norm):
    """ Euclidean distance.
    Args:
        feat:       N x D feature matrix
        query:      1 x D feature vector
        featl2norm: 1 x N vector

    Returns:
        dist:       1 x N vector
    """
    N = feat.shape[0]
    D = feat.shape[1]
    feat = np.require(feat, dtype=np.float32)
    query = np.require(query, dtype=np.float32)
    dist = np.empty((1, N), dtype=np.float32)
    # dist = featl2norm.astype(np.float32)
    fast_euclidean_core_cfunc(<cnp.float32_t *> cnp.PyArray_DATA(feat),
                              <cnp.float32_t *> cnp.PyArray_DATA(query),
                              <cnp.float32_t *> cnp.PyArray_DATA(featl2norm),
                              D, N,
                              <cnp.float32_t *> cnp.PyArray_DATA(dist))
    return dist
