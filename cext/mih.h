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

int get_keys_dist(uint32_t slice, uint32_t len, uint32_t dist, uint32_t * keys);


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
    int append(IDType id) {
      if (next_ >= cap_) {
        cap_ += step_;
        IDType * tmp = ids_;
        ids_ = new IDType[cap_];
        if (tmp != NULL) {
          memcpy(ids_, tmp, sizeof(ids_[0]) * next_);
        }
      }
      ids_[next_] = id;
      next_++;
      return 0;
    }

    int size() {
      return next_;
    }

    IDType get(int i) {
      return ids_[i];
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
    int search(uint8_t * query, uint32_t * ids, uint16_t * dis, int topk) const;

    int load(const char * idx_path);
    int save(const char * idx_path) const;

  protected:
    int nbits_;
    int code_len_;        // code length in bytes
    int sublen_;          // length of sub-code in bits
    int ntbls_;
    int nbkts_;
    Bucket<uint32_t> ** tables_;
    Bucket<uint32_t> * buckets_;

    uint8_t * codes_;
    size_t ncodes_;
    size_t capacity_;
    uint32_t * key_map_;
    vector<int> key_start_;
    vector<int> key_end_;
    uint8_t * bitmap_;
};


#endif
