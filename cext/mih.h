/*************************************************************************
  > File Name: mih.h
  > Copyright (C) 2013 Wan Ji<wanji@live.com>
  > Created Time: Wed 05 Nov 2014 02:38:01 PM CST
  > Descriptions: 
 ************************************************************************/

#ifndef _MIH_H_
#define _MIH_H_

#include <stdlib.h>
#include <string.h>
#include <vector>
#include <iostream>
using namespace std;

#include "common.h"

int get_keys_dist(uint32_t slice, int len, int dist, uint32_t * keys);


template <typename IDType>
class Bucket {
  public:
    Bucket(int step=2, int cap=0) : step_(step), cap_(cap), next_(0) {
      if (cap_ > 0) {
        ids_ = new IDType[cap_];
      } else {
        ids_ = NULL;
      }
    }

    ~Bucket() {
    }

    int reserve(int cap) {
      if (cap <= cap_) {
        return 0;
      }

      cap_ = cap;
      IDType * tmp = ids_;
      ids_ = new IDType[cap_];
      if (tmp != NULL) {
        memcpy(ids_, tmp, sizeof(ids_[0]) * next_);
        delete [] tmp;
      }

      return 0;
    }

    int append(IDType id) {
      if (next_ >= cap_) {
        reserve(cap_ + step_);
      }
      ids_[next_] = id;
      next_++;
      return 0;
    }

    const int & size() const {
      return next_;
    }

    int & size() {
      return next_;
    }

    const IDType & get(int i) const {
      return ids_[i];
    }

    IDType & get(int i) {
      return ids_[i];
    }

    const IDType * & ids() const {
      return ids_;
    }

    IDType * & ids() {
      return ids_;
    }

  protected:
    int step_;
    int cap_;
    int next_;
    IDType * ids_;
};

class MultiIndexer {
  public:
    MultiIndexer(int nbits, int ntbls=1, int capacity=0);
    ~MultiIndexer();
    int get_num_items() { return ncodes_; }

    int add(uint8_t * codes, int num);
    int search(uint8_t * query, int32_t * ids, int16_t * dis, int topk) const;

    int load(const char * idx_path);
    int save(const char * idx_path) const;

  protected:
    int code_len_;        // code length in bytes
    int sublen_;          // length of sub-code in bits
    uint32_t nbits_;
    uint32_t ntbls_;
    uint32_t nbkts_;
    Bucket<uint32_t> ** tables_;
    Bucket<uint32_t> * buckets_;

    uint8_t * codes_;
    uint64_t ncodes_;
    uint64_t capacity_;
    uint32_t * key_map_;
    vector<int> key_start_;
    vector<int> key_end_;
    uint8_t * bitmap_;
};


#endif
