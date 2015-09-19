/*************************************************************************
  > File Name: cext.h
  > Copyright (C) 2013 Wan Ji<wanji@live.com>
  > Created Time: Wed 05 Nov 2014 02:38:01 PM CST
  > Descriptions: 
 ************************************************************************/

#ifdef _WIN32
typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int   uint32_t;
typedef unsigned long  uint64_t;

typedef char  int8_t;
typedef short int16_t;
typedef int   int32_t;
typedef long  int64_t;
#else
#include <stdint.h>
#endif


void sumidxtab_core_cfunc(const float * D, const uint8_t * blk,
    int nsq, int ksub, int cur_num, float * out);

void hamming_core_cfunc(uint8_t * db, uint8_t * qry, int dim, int num,
    uint16_t * dist);

void knn_count_core_cfunc(const uint16_t * D, int numD, int maxD,
    int topk, int32_t * out);

void fast_euclidean_core_cfunc(const float * feat, const float * query,
    const float * featl2norm, int dim, int num,
    float * dist);
