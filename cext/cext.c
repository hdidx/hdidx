/*************************************************************************
  > File Name: cext.c
  > Copyright (C) 2013 Wan Ji<wanji@live.com>
  > Created Time: Wed 05 Nov 2014 03:46:12 PM CST
  > Descriptions: 
 ************************************************************************/

#define DBG_TIME 0

#include "cext.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if DBG_TIME
#include <sys/time.h>
#endif

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

#if DBG_TIME
  struct timeval t0, t1;
  gettimeofday(&t0, NULL);
#endif
  for (i=0; i<cur_num; i++) {
    out[i] = 0.0;
    const uint8_t * prow = blk + nsq * i;
    for (j=0; j<nsq; j++) {
      out[i] += D[j * ksub + prow[j]];
    }
  }
#if DBG_TIME
  gettimeofday(&t1, NULL);
  fprintf(stderr, "%.4f\n", (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) / 1000000.0);
#endif
}

int pq_knn_cfunc_partition(const float * dist, int start, int end, int32_t * idxs) {
  float x = dist[idxs[start]];
  int32_t q = start;
  int32_t i = start + 1;
  int32_t tmp;
  // fprintf(stderr, "start: %d, end: %d\n", start, end);
  while (i < end) {
    // fprintf(stderr, "i: %d, q: %d - begin\n", i, q);
    if (dist[idxs[i]] <= x) {
      q++;
      tmp = idxs[q]; idxs[q] = idxs[i]; idxs[i] = tmp;
    }
    // fprintf(stderr, "i: %d, q: %d - end\n", i, q);
    i++;
  }
  tmp = idxs[q]; idxs[q] = idxs[start]; idxs[start] = tmp;
  // fprintf(stderr, "start: %d, end: %d\n", start, end);
  return q;
}

void pq_knn_cfunc_sort(const float * dist, int32_t start, int32_t end, int32_t * idxs) {
  if (start >= end) {
    return;
  }

  int32_t q = pq_knn_cfunc_partition(dist, start, end, idxs);
  pq_knn_cfunc_sort(dist, start, q, idxs);
  pq_knn_cfunc_sort(dist, q+1, end, idxs);
}

void pq_knn_cfunc(const float * dist, int32_t total, int32_t topk, int32_t * out) {
  int32_t i;

  for (i=0; i<total; i++) {
    out[i] = i;
  }

  int32_t start = 0;
  int32_t end = total;
  int32_t q;
  // fprintf(stderr, "start: %d, end: %d\n", start, end);
  while ((q = pq_knn_cfunc_partition(dist, start, end, out)) != topk) {
    if (q > topk) {
      end = q;
    } else {
      start = q > start ? q : start + 1;
    }
    // fprintf(stderr, "start: %d, end: %d\n", start, end);
  }

  // pq_knn_cfunc_sort(dist, 0, topk, out);
}
