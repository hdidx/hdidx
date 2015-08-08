/*************************************************************************
  > File Name: hamdist.hpp
  > Copyright (C) 2013 Wan Ji<wanji@live.com>
  > Created Time: Fri 27 Dec 2013 01:46:49 PM CST
  > Descriptions: 
 ************************************************************************/

#ifndef _HAMDIST_H_
#define _HAMDIST_H_

/**
 * number of `1`s in 0~255
 */
static const int BIT_CNT_MAP[] = {
  0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
  1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
  1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
  2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
  1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
  2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
  2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
  3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
  1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
  2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
  2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
  3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
  2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
  3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
  3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
  4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8};

/**
 * hamming distance of 32-bit codes
 */
inline int hamdist32(uint32_t a, uint32_t b) {
  return __builtin_popcount(a ^ b);
}

/**
 * hamming distance of 64-bit codes
 */
inline int hamdist64(uint64_t a, uint64_t b) {
  return __builtin_popcountll(a ^ b);
}

/**
 * hamming distance of codes with arbitrary length
 */
static inline int hamdist(const uint8_t * a, const uint8_t * b, int nbytes) {
  int dist = 0;
  int pos  = 8;
  switch (nbytes) {
    case 4:   // 32-bit
      dist = hamdist32(*(uint32_t *)a, *(uint32_t *)b);
      break;
    case 8:   // 64-bit
      dist = hamdist64(*(uint64_t *)a, *(uint64_t *)b);
      break;
    case 16:  // 128-bit
      dist = hamdist64(((uint64_t *)a)[0], ((uint64_t *)b)[0]) +
             hamdist64(((uint64_t *)a)[1], ((uint64_t *)b)[1]);
      break;
    case 32:  // 256-bit
      dist = hamdist64(((uint64_t *)a)[0], ((uint64_t *)b)[0]) +
             hamdist64(((uint64_t *)a)[1], ((uint64_t *)b)[1]) +
             hamdist64(((uint64_t *)a)[2], ((uint64_t *)b)[2]) +
             hamdist64(((uint64_t *)a)[3], ((uint64_t *)b)[3]);
      break;
    default:  // arbitrary length
      /* calc 64-bit substrings first */
      while (pos < nbytes) {
        dist += hamdist64(*(uint64_t *)a, *(uint64_t *)b);
        a += 8; b += 8; pos += 8;
      }
      /* then 32-bit substrings */
      pos -= (8-4);
      while (pos < nbytes) {
        dist += hamdist32(*(uint32_t *)a, *(uint32_t *)b);
        a += 4; b += 4; pos += 4;
      }
      /* then substrings shorter than 32-bit */
      pos -= (4-1);
      while (pos < nbytes) {
        dist += BIT_CNT_MAP[(*a) ^ (*b)];
        a++; b++; pos++;
      }
  }
  return dist;
}

#endif

