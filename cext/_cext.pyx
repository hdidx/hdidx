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
    void hamming_core_cfunc(cnp.uint8_t * db, cnp.uint8_t * qry, int dim, int num,
                            cnp.uint16_t * dist)

# create the wrapper code, with numpy type annotations
def sumidxtab_core(cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] D not None,
                   cnp.ndarray[cnp.uint8_t, ndim=2, mode="c"] blk not None):
    out = np.zeros(blk.shape[0], dtype=D.dtype)
    sumidxtab_core_cfunc(<cnp.float32_t*> cnp.PyArray_DATA(D),
                         <cnp.uint8_t*> cnp.PyArray_DATA(blk),
                         D.shape[0], D.shape[1], blk.shape[0],
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
                           dq, nd,
                           <cnp.uint16_t*> cnp.PyArray_DATA(out[i, :]))
    return out
