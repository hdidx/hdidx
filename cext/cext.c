/*************************************************************************
  > File Name: cext.c
  > Copyright (C) 2013 Wan Ji<wanji@live.com>
  > Created Time: Wed 05 Nov 2014 03:46:12 PM CST
  > Descriptions: 
 ************************************************************************/

#include "cext.h"
#include "hamdist.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>


void hamming_core_cfunc(uint8_t * qry, uint8_t * db, int dim, int num,
    uint16_t * dist) {
  int i;
  uint8_t * pdb;
  for (i=0, pdb=db; i<num; i++, pdb+=dim) {
    dist[i] = hamdist(qry, pdb, dim);
  }
}

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

/**
 * kNN for Hamming distance based on counting sort.
 */
void knn_count_core_cfunc(const uint16_t * D, int numD, int maxD,
    int topk, int32_t * out) {
  int * counter = (int *)malloc((maxD + 1) * sizeof(counter[0]));
  int * v_pos = (int *)malloc((maxD + 1) * sizeof(counter[0]));
  memset(counter, 0, (maxD + 1) * sizeof(counter[0]));
  int i, pos;
  for (i=0; i<numD; i++) {
    counter[D[i]]++;
  }
  v_pos[0] = 0;
  for (i=0; i<maxD; i++) {
    v_pos[i+1] = v_pos[i] + counter[i];
  }
  for (i=0; i<numD; i++) {
    pos = v_pos[D[i]];
    if (pos >= topk) {
      continue;
    }
    out[pos] = i;
    v_pos[D[i]]++;
  }
  free(counter);
  free(v_pos);
}
