/*************************************************************************
  > File Name: cext.h
  > Copyright (C) 2013 Wan Ji<wanji@live.com>
  > Created Time: Wed 05 Nov 2014 02:38:01 PM CST
  > Descriptions: 
 ************************************************************************/

#ifndef _CEXT_H_
#define _CEXT_H_

#include "common.h"

void sumidxtab_core_cfunc(const float * D, const uint8_t * blk,
    int nsq, int ksub, int cur_num, float * out);

void hamming_core_cfunc(uint8_t * db, uint8_t * qry, int dim, int num,
    uint16_t * dist);

void knn_count_core_cfunc(const uint16_t * D, int numD, int maxD,
    int topk, int32_t * out);

void fast_euclidean_core_cfunc(const float * feat, const float * query,
    const float * featl2norm, int dim, int num,
    float * dist);

#endif
