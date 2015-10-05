/*************************************************************************
  > File Name: mih.c
  > Copyright (C) 2013 Wan Ji<wanji@live.com>
  > Created Time: Fri 02 Oct 2015 03:51:12 PM CST
  > Descriptions: 
 ************************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <algorithm>
#include <assert.h>
#include <iostream>
#include <set>

using namespace std;

#include "mih.h"
#include "hamdist.h"


int get_keys_dist(uint32_t slice, int len, int dist, uint32_t * keys) {
  int * flags = new int[len];
  int count = 0;
  for (int32_t j=0; j<dist; j++) {
    flags[j] = 0;
  }
  for (int32_t j=dist; j<len; j++) {
    flags[j] = 1;
  }
  do {
    int32_t key = slice;
    for (int32_t k = 0; k < len; ++k) {
      if (!flags[k]) {
        key ^= 1<<k;
      }
    }
    keys[count++] = key;
  } while (std::next_permutation(flags, flags+len));
  delete [] flags;
  return count;
}

/**
 * Get substring from `bits`，pos/len indicate bits, not bytes.
 */
uint32_t subits(const uint8_t * bits, int pos, int len) {
  assert(len <= 32);
  uint32_t sub;
  int sq = pos / 8;
  int sr = pos % 8;
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

  return sub;
}

MultiIndexer::MultiIndexer(int nbits, int ntbls, int capacity) :
  nbits_(nbits), ntbls_(ntbls), capacity_(capacity) {

  bitmap_ = NULL;

  code_len_ = (nbits_ + 7) / 8;

  nbkts_ = 1;
  for (uint32_t i=0; i<nbits_ / ntbls_; i++) {
    nbkts_ *= 2;
  }
  tables_ = new Bucket<uint32_t> *[ntbls_];
  buckets_ = new Bucket<uint32_t>[nbkts_ * ntbls_];

  for (int i=0; i<ntbls; i++) {
    tables_[i] = buckets_ + i * nbkts_;
  }

  ncodes_ = 0;
  if (capacity_ > 0) {
    codes_ = new uint8_t[capacity_ * code_len_];
  } else {
    codes_ = NULL;
  }

  sublen_ = nbits_ / ntbls_;
  // this restriction is not necessary, and will be changed in the future
  assert(nbits_ % ntbls_ == 0);

  key_map_ = new uint32_t[nbkts_];
  int start = 0;
  int end = 0;
  for (int i=0; i<=sublen_; i++) {
    int num = get_keys_dist(0, sublen_, i, key_map_ + start);
    end += num;
    key_start_.push_back(start);
    key_end_.push_back(end);
    start += num;
  }
}

MultiIndexer::~MultiIndexer() {
  if (tables_ != NULL) {
    delete [] tables_;
  }
  if (buckets_ != NULL) {
    delete [] buckets_;
  }
  if (codes_ != NULL) {
    delete [] codes_;
  }
  if (key_map_ != NULL) {
    delete [] key_map_;
  }
  if (bitmap_ != NULL) {
    delete [] bitmap_;
  }
}

int MultiIndexer::add(uint8_t * codes, int num) {
  /**
   * Append codes to the end of codes_
   */
  if (ncodes_ + num > capacity_) {
    capacity_ = ncodes_ + num;
    uint8_t * tmp = codes_;
    codes_ = new uint8_t[capacity_ * code_len_];
    if (tmp != NULL) {
      memcpy(codes_, tmp, ncodes_ * code_len_);
      delete [] tmp;
    }
  }
  memcpy(codes_ + ncodes_ * code_len_, codes, num * code_len_);

  /**
   * Insert codes into each hash table
   */
  for (uint32_t id=ncodes_; id<ncodes_+num; id++) {
    uint8_t * code = codes_ + id * code_len_;
    for (uint32_t i=0; i<ntbls_; i++) {
      uint32_t subkey = subits(code, sublen_ * i, sublen_);
      tables_[i][subkey].append(id);
    }
  }

  ncodes_ += num;
  if (bitmap_ != NULL) {
    delete [] bitmap_;
  }
  bitmap_ = new uint8_t[ncodes_];
  return 0;
}

int MultiIndexer::search(uint8_t * query, int32_t * ids, int16_t * dis, int topk) const {
  vector<vector<uint32_t> > v_ret(nbits_+1);
  int sublen_ = nbits_ / ntbls_;

  int acc = 0;
  int last_sub_dist = -1;
  memset(bitmap_, 0, ncodes_);
  // search for different distance
  for (uint32_t d=0; d<=nbits_; d++) {
    int sub_dist = d / ntbls_;
    // only update v_ret when sub_dist chaged
    if (sub_dist != last_sub_dist) {
      // scan the tables
      for (uint32_t i=0; i<ntbls_; i++) {
        uint32_t subcode = subits(query, sublen_ * i, sublen_);
        // scan the buckets with distance `sub_dist` in each table
        for (int t=key_start_[sub_dist]; t<key_end_[sub_dist]; t++) {
          Bucket<uint32_t> & bucket = tables_[i][key_map_[t] ^ subcode];
          for (int j=0; j<bucket.size(); j++) {
            uint32_t id = bucket.get(j);
            if (bitmap_[id]) {
              continue;
            }
            bitmap_[id] = 1;
            uint16_t dist = hamdist(query, codes_ + id * code_len_, code_len_);
            v_ret[dist].push_back(id);
          }
        }
      }
    }

    for (uint32_t i=0; i<v_ret[d].size() && acc < topk; i++, acc++) {
      *ids++ = v_ret[d][i];
      *dis++ = d;
    }
    if (acc >= topk) {
      break;
    }
    last_sub_dist = sub_dist;
  }
  return 0;
}

int MultiIndexer::load(const char * idx_path) {

  FILE * fp = fopen(idx_path, "rb");
  if (fp == NULL) {
    fprintf(stderr, "Error: cannot open file %s for writing!\n", idx_path);
    return -1;
  }

  int rdcnt = 0;
  uint64_t rsrved;
  uint32_t ntbls, nbits;
  // skip 8 bytes reserved field
  rdcnt += fread(&rsrved, sizeof(rsrved), 1, fp);
  // load number of tables/bits/codes
  rdcnt += fread(&ntbls, sizeof(ntbls), 1, fp);
  rdcnt += fread(&nbits, sizeof(nbits), 1, fp);
  rdcnt += fread(&ncodes_, sizeof(ncodes_), 1, fp);
  // skip 8 bytes reserved field
  rdcnt += fread(&rsrved, sizeof(rsrved), 1, fp);
  assert(nbits_ == nbits);
  assert(ntbls_ == ntbls);

  /**
   * load binary codes
   */
  if (codes_ != NULL) {
    delete [] codes_;
  }
  capacity_ = ncodes_;
  codes_ = new uint8_t[ncodes_ * code_len_];
  rdcnt = fread(codes_, code_len_, ncodes_, fp);
  fprintf(stderr, "\t%d codes loaded!\n", rdcnt);
  /**
   * init bitmap
   */
  if (bitmap_ != NULL) {
    delete [] bitmap_;
  }
  bitmap_ = new uint8_t[ncodes_];

  /**
   * load buckets
   */
  uint64_t bid;
  int bucket_size;
  rdcnt = fread(&bid, sizeof(bid), 1, fp);
  while (bid != 0xffffffffffffffff) {
    rdcnt += fread(&bucket_size, sizeof(bucket_size), 1, fp);
    buckets_[bid].reserve(bucket_size);
    // if (bid % 1000 == 0) 
    //   fprintf(stderr, "bid/bcnt: %lu\t%d\n", bid, bucket_size);
    rdcnt += fread(buckets_[bid].ids(), sizeof(buckets_[bid].ids()[0]),
        bucket_size, fp);
    buckets_[bid].size() = bucket_size;
    rdcnt += fread(&bid, sizeof(bid), 1, fp);
  }

  fclose(fp);
  return 0;
}

int MultiIndexer::save(const char * idx_path) const {
  FILE * fp = fopen(idx_path, "wb");
  if (fp == NULL) {
    fprintf(stderr, "Error: cannot open file %s for writing!\n", idx_path);
    return -1;
  }

  uint64_t rsrved = 0xffffffffffffffff;
  // write 8 bytes reserved field
  fwrite(&rsrved, sizeof(rsrved), 1, fp);
  // write number of tables/bits/codes
  fwrite(&ntbls_, sizeof(ntbls_), 1, fp);
  fwrite(&nbits_, sizeof(nbits_), 1, fp);
  fwrite(&ncodes_, sizeof(ncodes_), 1, fp);
  // write 8 bytes reserved field
  rsrved = 0;
  fwrite(&rsrved, sizeof(rsrved), 1, fp);

  /**
   * dump binary codes
   */
  int wrcnt = fwrite(codes_, code_len_, ncodes_, fp);
  fprintf(stderr, "\t%d codes saved!\n", wrcnt);

  /**
   * dump buckets
   */
  uint64_t bucket_num = nbkts_ * ntbls_;
  for (uint64_t bid=0; bid<bucket_num; bid++) {
    if (buckets_[bid].size() > 0) {
      // if (bid % 1000 == 0) 
      //   fprintf(stderr, "bid/bcnt: %lu\t%d\n", bid, buckets_[bid].size());
      fwrite(&bid, sizeof(bid), 1, fp);
      fwrite(&(buckets_[bid].size()), sizeof(buckets_[bid].size()), 1, fp);
      fwrite(buckets_[bid].ids(), sizeof(buckets_[bid].ids()[0]), buckets_[bid].size(), fp);
    }
  }
  uint64_t endid = 0xffffffffffffffff;
  fwrite(&endid, sizeof(endid), 1, fp);

  fclose(fp);
  return 0;
}
