/*************************************************************************
  > File Name: mih.c
  > Copyright (C) 2013 Wan Ji<wanji@live.com>
  > Created Time: Fri 02 Oct 2015 03:51:12 PM CST
  > Descriptions: 
 ************************************************************************/

#include "mih.h"
#include "hamdist.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <algorithm>
#include <assert.h>

using namespace std;


int get_keys_dist(uint32_t slice, uint32_t len, uint32_t dist, uint32_t * keys) {
  int flags[len];
  int count = 0;
  for (uint32_t j=0; j<dist; j++) {
    flags[j] = 0;
  }
  for (uint32_t j=dist; j<len; j++) {
    flags[j] = 1;
  }
  do {
    uint32_t key = slice;
    for (uint32_t k = 0; k < len; ++k) {
      if (!flags[k]) {
        key ^= 1<<k;
      }
    }
    keys[count++] = key;
  } while (std::next_permutation(flags, flags+len));
  return count;
}

/**
 * Get substring from `bits`，pos/len indicate bits, not bytes.
 */
int subits(const uint8_t * bits, uint32_t & sub, int pos, int len) {
  assert(len <= 32);
  
  int sq = pos/8;
  int sr = pos%8;
  int eq = (pos + len)/8;
  /* Beginning byte */
  int sb = sq;
  /* Ending byte */
  int eb = eq;

  sub = *(uint32_t *)(bits+sb);
  int restlen = 32 - len - sr;
  /**
   * restlen >= 0 means sub falls in one 32bit integer 
   * otherwise, sub crosses two 32bit integer
   */
  if (restlen >= 0) {
    sub = sub << restlen;
    sub = sub >> restlen;
    sub = sub >> sr;
  } else {
    sub = sub >> sr;
    uint32_t rest = (uint32_t)bits[eb];
    rest = rest << (len + restlen);
    sub |= rest;
  }

  return 0;
}
