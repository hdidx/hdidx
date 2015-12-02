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
from libc.stdint cimport int8_t, uint8_t, \
    int16_t, uint16_t, int32_t, uint32_t
import operator
from itertools import izip as zip

from hdidx import _cext as cext
from functools import reduce

# use the Numpy-C-API from Cython
cnp.import_array()

# cdefine the signature of our c function
cdef extern from "mih.h":
    int get_keys_dist(uint32_t slice, int len,
                      int dist, uint32_t * keys)

cdef extern from "hamdist.h":
    uint16_t hamdist(uint8_t * qry, uint8_t * db, int dim)


def c(n,k):
    return reduce(operator.mul, range(n - k + 1, n + 1)) /\
        reduce(operator.mul, range(1, k +1)) if k > 0 else 1

# create the wrapper code, with numpy type annotations
def get_key_map(nbits):
    if nbits > 32:
        raise Exception("Number of bits cannot exceed 32!")
    out = []
    for d in xrange(nbits+1):
        codes = np.zeros(c(nbits, d), dtype=np.uint32)
        num = get_keys_dist(0, nbits, d,
                            <uint32_t *> cnp.PyArray_DATA(codes))
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

def search(qry_code, db_codes, tables, key_map, topk):
    pass

def search_for_sub_dist(sub_dist, qry_keys, qry_code, proced,
                        db_codes, idmap,
                        tables, key_map, ret_set):
    nbits = qry_code.shape[-1]
    for table, subcode in zip(tables, qry_keys[0]):
        for mask in key_map[sub_dist]:
            for cur_id in table.get(mask ^ subcode, []):
                # if not proced.test(cur_id):
                #     proced.set(cur_id)
                if cur_id not in proced:
                    proced.add(cur_id)
                    dist = hamdist(<uint8_t *> cnp.PyArray_DATA(qry_code),
                                   <uint8_t *> cnp.PyArray_DATA(db_codes[idmap[cur_id]]),
                                   nbits)
                    ret_set[dist].append(cur_id)

cdef extern from "mih.h":
    cdef cppclass MultiIndexer:
        MultiIndexer(int, int, int)
        int get_num_items()
        int add(uint8_t * codes, int num)
        int search(uint8_t * query, int32_t * ids,
                   int16_t * dis, int topk)
        int load(char * codes)
        int save(char * codes)

cdef class PyMultiIndexer:
    cdef MultiIndexer * thisptr

    def __cinit__(self, int nbits, int ntbls, int capacity=0):
        self.thisptr = new MultiIndexer(nbits, ntbls, capacity)

    def __dealloc__(self):
        del self.thisptr

    def get_num_items(self):
        return self.thisptr.get_num_items()

    def add(self, cnp.ndarray[uint8_t, ndim=2, mode="c"] codes):
        return self.thisptr.add(<uint8_t *> cnp.PyArray_DATA(codes),
                                codes.shape[0])

    def search(self, cnp.ndarray[uint8_t, ndim=2, mode="c"] qry, int topk):
        ids = np.zeros(topk, np.int32)
        dis = np.zeros(topk, np.int16)
        self.thisptr.search(<uint8_t *> cnp.PyArray_DATA(qry),
                            <int32_t *> cnp.PyArray_DATA(ids),
                            <int16_t *> cnp.PyArray_DATA(dis),
                            topk)
        return ids, dis

    def load(self, idx_path):
        return self.thisptr.load(<char *> idx_path)

    def save(self, idx_path):
        return self.thisptr.save(<char *> idx_path)
