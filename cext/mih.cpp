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

  code_len_ = (nbits_ + 7) / 8;

  nbkts_ = 1;
  for (int i=0; i<nbits_ / ntbls_; i++) {
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
  vector<int> count;
  for (int i=0; i<ntbls_; i++) {
    count.push_back(0);
  }
  for (uint32_t id=ncodes_; id<ncodes_+num; id++) {
    uint8_t * code = codes_ + id * code_len_;
    // for (int i=0; i<code_len_; i++) {
    //   printf("%d\t", code[i]);
    // }
    // cout << endl;
    for (int i=0; i<ntbls_; i++) {
      uint32_t subkey = subits(code, sublen_ * i, sublen_);
      // cout << subkey << '\t';
      if (subkey == 0) {
        count[i]++;
      }
      tables_[i][subkey].append(id);
    }
  }
  for (int i=0; i<ntbls_; i++) {
    cout << count[i] << '\t';
  }
  cout << endl;

  ncodes_ += num;
}

int MultiIndexer::search(uint8_t * query, uint32_t * ids, uint16_t * dis, int topk) const {
  set<uint32_t> proced;
  int sublen_ = nbits_ / ntbls_;
  vector<uint32_t> subcodes;
  for (int i=0; i<ntbls_; i++) {
    subcodes.push_back(subits(query, sublen_ * i, sublen_));
  }

  vector<vector<uint16_t> > v_ret;
  for (int d=0; d<=nbits_; d++) {
    v_ret.push_back(vector<uint16_t>());
  }

  int acc = 0;
  int last_sub_dist = -1;
  for (int d=0; d<=nbits_; d++) {
    int sub_dist = d / ntbls_;
    if (sub_dist != last_sub_dist) {
      for (int i=0; i<ntbls_; i++) {
        for (int t=key_start_[sub_dist]; t<key_end_[sub_dist]; t++) {
          uint32_t sub_key = key_map_[t] ^ subcodes[i];
          for (int j=0; j<tables_[i][sub_key].size(); j++) {
            uint32_t id = tables_[i][sub_key].get(j);
            if (proced.find(id) != proced.end()) {
              continue;
            }
            proced.insert(id);
            uint16_t dist = hamdist(query, codes_ + id * code_len_, code_len_);
            v_ret[dist].push_back(id);
          }
        }
      }
    }

    for (int i=0; i<v_ret[d].size() && acc < topk; i++, acc++) {
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
  cout << "load from " << idx_path << endl;
  return 0;
}

int MultiIndexer::save(const char * idx_path) const {
  cout << "save to " << idx_path << endl;
  return 0;
}
