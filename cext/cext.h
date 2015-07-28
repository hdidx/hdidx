/*************************************************************************
  > File Name: cext.h
  > Copyright (C) 2013 Wan Ji<wanji@live.com>
  > Created Time: Wed 05 Nov 2014 02:38:01 PM CST
  > Descriptions: 
 ************************************************************************/
#include <stdint.h>

void sumidxtab_core_cfunc(const float * D, const uint8_t * blk,
    int nsq, int ksub, int cur_num, float * out);

void hamming_core_cfunc(uint8_t * db, uint8_t * qry, int dim, int num,
    uint16_t * dist);

