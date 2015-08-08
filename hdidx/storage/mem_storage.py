#!/usr/bin/env python
# coding: utf-8

"""
   File Name: mem_storage.py
      Author: Wan Ji
      E-mail: wanji@live.com
  Created on: Sat Aug  8 16:11:01 2015 CST
"""
DESCRIPTION = """
"""

import itertools
import numpy as np
from hdidx.storage import Storage


class MemStorage(Storage):
    def __init__(self):
        Storage.__init__(self)
        self.keys = np.arange(0, dtype=np.int32)
        self.codes = None
        self.num_items = 0

    def add(self, codes, keys):
        num_new_items = codes.shape[0]
        self.keys = np.hstack((self.keys, keys))
        if self.codes is None:
            self.codes = codes
        else:
            self.codes = np.vstack((self.codes, codes))
        self.num_items += num_new_items
        return num_new_items

    def __iter__(self):
        return itertools.izip(self.keys, self.vals)
