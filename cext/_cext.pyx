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
cimport numpy as np

# use the Numpy-C-API from Cython
np.import_array()

# cdefine the signature of our c function
cdef extern from "cext.h":
    void sumidxtab_core_cfunc(const np.float32_t * D, const np.uint8_t * blk,
        int nsq, int ksub, int cur_num, np.float32_t * out)

# create the wrapper code, with numpy type annotations
def sumidxtab_core(np.ndarray[np.float32_t, ndim=2, mode="c"] D not None,
                   np.ndarray[np.uint8_t, ndim=2, mode="c"] blk not None):
    out = np.zeros(blk.shape[0], dtype=D.dtype)
    sumidxtab_core_cfunc(<np.float32_t*> np.PyArray_DATA(D),
                         <np.uint8_t*> np.PyArray_DATA(blk),
                         D.shape[0], D.shape[1], blk.shape[0],
                         <np.float32_t*> np.PyArray_DATA(out))
    return out
