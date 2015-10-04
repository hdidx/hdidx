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
from itertools import izip as zip

from hdidx import _cext as cext

# use the Numpy-C-API from Cython
cnp.import_array()

# cdefine the signature of our c function
cdef extern from "mih.h":
    int get_keys_dist(cnp.uint32_t slice, cnp.uint32_t len,
                      cnp.uint32_t dist,  cnp.uint32_t * keys);

cdef extern from "hamdist.h":
    cnp.uint16_t hamdist(cnp.uint8_t * qry, cnp.uint8_t * db, int dim)


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
                    dist = hamdist(<cnp.uint8_t *> cnp.PyArray_DATA(qry_code),
                                   <cnp.uint8_t *> cnp.PyArray_DATA(db_codes[idmap[cur_id]]),
                                   nbits)
                    ret_set[dist].append(cur_id)


def filtering(sub_dist, qry_keys, proced, tables, key_map):
    return _filtering(sub_dist, qry_keys, proced, tables, key_map)

cdef _filtering(sub_dist, qry_keys, proced, tables, key_map):
    # candidates = reduce(operator.concat, [
    #     reduce(operator.concat, [
    #         table.get(mask ^ subcode, [])
    #         for mask in key_map[sub_dist]
    #     ]) for table, subcode in zip(tables, qry_keys[0])
    # ])

    # filtered = []
    # for cur_id in candidates:
    #     if cur_id not in proced:
    #         filtered.append(cur_id)
    #         proced.add(cur_id)

    filtered = []
    for table, subcode in zip(tables, qry_keys[0]):
        for mask in key_map[sub_dist]:
            for cur_id in table.get(mask ^ subcode, []):
                # if not proced.test(cur_id):
                #     filtered.append(cur_id)
                #     proced.set(cur_id)
                if cur_id not in proced:
                    filtered.append(cur_id)
                    proced.add(cur_id)
    return filtered

def fill_table(qry_code, candidates, candidates_ids, ret_set):
    cur_dist = cext.hamming(qry_code, candidates).reshape(-1)
    # print sub_dist, sorted(zip(cur_dist, filtered),
    #                     key=lambda x: x[0])[:10]
    for i in xrange(len(candidates_ids)):
        ret_set[cur_dist[i]].append(candidates_ids[i])
