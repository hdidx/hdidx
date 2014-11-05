/*************************************************************************
  > File Name: cext.c
  > Copyright (C) 2013 Wan Ji<wanji@live.com>
  > Created Time: Wed 05 Nov 2014 03:46:12 PM CST
  > Descriptions: 
 ************************************************************************/

#include "cext.h"
#include <stdio.h>

/**
 * D:   nsq * ksub
 * blk: cur_num * nsq
 */
void sumidxtab_core_cfunc(const float * D, const uint8_t * blk,
    int nsq, int ksub, int cur_num, float * out) {
  int i, j;

  /*
  for (i=0; i<nsq; i++) {
    const float * pD = D + ksub * i;
    for (j=0; j<ksub; j++) {
      fprintf(stderr, "%8.4f", pD[j]);
    }
    fprintf(stderr, "\n");
  }

  for (i=0; i<cur_num; i++) {
    const uint8_t * pblk = blk + nsq * i;
    for (j=0; j<nsq; j++) {
      fprintf(stderr, "%8u", pblk[j]);
    }
    fprintf(stderr, "\n");
  }
  */

  for (i=0; i<cur_num; i++) {
    out[i] = 0.0;
    const uint8_t * prow = blk + nsq * i;
    for (j=0; j<nsq; j++) {
      out[i] += D[j * ksub + prow[j]];
    }
  }
}
