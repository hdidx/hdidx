#!/usr/bin/env python
# coding: utf-8

#########################################################################
#########################################################################

"""
   File Name: storage
      Author: Wan Ji
      E-mail: wanji@live.com
  Created on: Tue Nov  4 11:25:03 2014 CST
"""
DESCRIPTION = """
"""

import lmdb


class Storage(object):
    def __init__(self):
        self.keys = None
        self.codes = None
        self.num_emptys = -1
        self.num_items = -1

    def get_num_items(self):
        return self.num_items

    def get_num_emptys(self):
        return self.num_emptys

    def get_keys(self):
        return self.keys

    def get_codes(self):
        return self.codes

    def __iter__(self):
        raise Exception("Instance of `Storage` is not allowed!")

    def next(self):
        raise Exception("Instance of `Storage` is not allowed!")

    def clear(self):
        raise Exception("Instance of `Storage` is not allowed!")


from mem_storage import MemStorage
from lmdb_storage import LMDBStorage


STORAGE_DIC = {
    'mem':  MemStorage,
    'lmdb': LMDBStorage,
}


def createStorage(storage_type, storage_parm=None):
    num_idx = storage_parm.get('num_idx', 0) if storage_parm else 0

    if storage_type == 'mem':
        # create inverted file storage
        if num_idx > 0:
            return [MemStorage() for i in xrange(num_idx)]
        # create normal storage
        else:
            return MemStorage()
    elif storage_type == 'lmdb':
        path = storage_parm['path']
        clear = storage_parm.get('clear', False)

        # create inverted file storage
        if num_idx > 0:
            env = lmdb.open(path, map_size=2**30, sync=False, max_dbs=num_idx)
            return [LMDBStorage(env, clear, i) for i in xrange(num_idx)]
        # create normal storage
        else:
            env = lmdb.open(path, map_size=2**30, sync=False, max_dbs=1)
            return LMDBStorage(env, clear)
    else:
        raise Exception('Wroing storage type: %s' % storage_type)
