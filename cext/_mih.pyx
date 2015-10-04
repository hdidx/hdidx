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
import operator

# use the Numpy-C-API from Cython
cnp.import_array()

# cdefine the signature of our c function
cdef extern from "mih.h":
    int get_keys_dist(cnp.uint32_t slice, cnp.uint32_t len,
                      cnp.uint32_t dist,  cnp.uint32_t * keys);


def c(n,k):
    return reduce(operator.mul, range(n - k + 1, n + 1)) /\
        reduce(operator.mul, range(1, k +1)) if k > 0 else 1

# create the wrapper code, with numpy type annotations
def get_key_map(nbits):
    if nbits > 32:
        raise StandardError("Number of bits cannot exceed 32!")
    out = []
    for d in xrange(nbits+1):
        codes = np.zeros(c(nbits, d), dtype=np.uint32)
        num = get_keys_dist(0, nbits, d,
                            <cnp.uint32_t*> cnp.PyArray_DATA(codes))
        if num != codes.shape[0]:
            raise Exception("Memory leak!!!")
        out.append(codes)
    return out

def get_keys_16bit(codes):
    out = np.zeros((codes.shape[0], codes.shape[1]/2), np.uint16)
    for i in xrange(codes.shape[0]):
        for j in xrange(0, codes.shape[1], 2):
            out[i, j/2] = codes[i, j] * 256 + codes[i, j+1]
    return out
