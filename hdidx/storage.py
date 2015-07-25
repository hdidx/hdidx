#!/usr/bin/env python
# coding: utf-8

#########################################################################
#########################################################################

"""
   File Name: storage.py
      Author: Wan Ji
      E-mail: wanji@live.com
  Created on: Tue Nov  4 11:25:03 2014 CST
"""
DESCRIPTION = """
"""

import itertools
from struct import pack, unpack
import lmdb
import cPickle as pickle
import numpy as np


class LMDBAccessor(object):
    def __init__(self, env, dbname):
        self.env = env
        self.db = env.open_db(dbname)

    def close(self):
        return
        self.db.close()

    def setkv(self, key, val, ktype, vtype):
        with self.env.begin(write=True) as txt:
            txt.cursor(self.db).put(pack(ktype, key), pack(vtype, key))

    def setk(self, key, val, ktype):
        with self.env.begin(write=True) as txt:
            txt.cursor(self.db).put(pack(ktype, key), val)

    def setv(self, key, val, vtype):
        with self.env.begin(write=True) as txt:
            txt.cursor(self.db).put(key, pack(vtype, val))

    def set(self, key, val):
        with self.env.begin(write=True) as txt:
            txt.cursor(self.db).put(key, val)

    def getkv(self, key, ktype, vtype):
        with self.env.begin(write=True) as txt:
            val = txt.cursor(self.db).get(pack(ktype, key))
            if val is None:
                return None
            return unpack(vtype, val)[0]

    def getk(self, key, ktype):
        with self.env.begin(write=True) as txt:
            return txt.cursor(self.db).get(pack(ktype, key))

    def getv(self, key, vtype):
        with self.env.begin(write=True) as txt:
            val = txt.cursor(self.db).get(key)
            if val is None:
                return None
            return unpack(vtype, val)[0]

    def get(self, key):
        with self.env.begin(write=True) as txt:
            return txt.cursor(self.db).get(key)

    def setkivi(self, key, val):
        self.setkv(key, val, 'i', 'i')

    def setki(self, key, val):
        self.setk(key, val, 'i')

    def setvi(self, key, val):
        self.setv(key, val, 'i')

    def getkivi(self, key):
        return self.getkv(key, 'i', 'i')

    def getki(self, key):
        return self.getk(key, 'i')

    def getvi(self, key):
        return self.getv(key, 'i')


class PQStorage(object):
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
        raise Exception("Instance of `PQStorage` is not allowed!")

    def next(self):
        raise Exception("Instance of `PQStorage` is not allowed!")

    def clear(self):
        raise Exception("Instance of `PQStorage` is not allowed!")


class MemPQStorage(PQStorage):
    def __init__(self):
        PQStorage.__init__(self)
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


class LMDBPQStorage(MemPQStorage):
    def __init__(self, env, clear, ivfidx=None):
        MemPQStorage.__init__(self)

        self.env = env
        dbname = 'db%06d' % ivfidx if ivfidx else 'db'
        self.db = LMDBAccessor(self.env, dbname)

        if clear:
            self.clear()

        with self.env.begin() as txn:
            self.num_items = txn.stat(self.db.db)['entries']
            if self.num_items > 0:
                cursor = txn.cursor(self.db.db)
                keys = []
                codes = []
                for key, code in cursor:
                    keys.append(unpack('i', key)[0])
                    codes.append(pickle.loads(code))
                self.keys = np.array(keys)
                self.codes = np.vstack(tuple(codes))

    def __del__(self):
        self.db.close()

    def add(self, codes, keys):
        num_new_items = MemPQStorage.add(self, codes, keys)
        for idx in xrange(num_new_items):
            self.db.setki(keys[idx], pickle.dumps(codes[idx], protocol=2))

    def clear(self):
        with self.env.begin(write=True) as txt:
            txt.drop(self.db.db, False)

"""
    def __iter__(self):
        self.iter_txt = self.env.begin()
        self.cursor_keys = self.iter_txt.cursor(self.db_keys.db)
        self.cursor_vals = self.iter_txt.cursor(self.db_vals.db)
        return self

    def next(self):
        status_keys = self.cursor_keys.next()
        status_vals = self.cursor_vals.next()
        if status_keys and status_vals:
            keys = pickle.loads(self.cursor_keys.value())
            vals = pickle.loads(self.cursor_vals.value())
            return keys, vals
        else:
            # self.iter_txt.close()
            raise StopIteration
"""


PQ_DIC = {
    'mem':  MemPQStorage,
    'lmdb': LMDBPQStorage,
}


def createStorage(storage_type, storage_parm=None):
    coarsek = storage_parm.get('coarsek', 0) if storage_parm else 0

    if storage_type == 'mem':
        # create inverted file storage
        if coarsek > 0:
            return [MemPQStorage() for i in xrange(coarsek)]
        # create normal storage
        else:
            return MemPQStorage()
    elif storage_type == 'lmdb':
        path = storage_parm['path']
        clear = storage_parm.get('clear', False)

        # create inverted file storage
        if coarsek > 0:
            env = lmdb.open(path, map_size=2**30, sync=False, max_dbs=coarsek)
            return [LMDBPQStorage(env, clear, i) for i in xrange(coarsek)]
        # create normal storage
        else:
            env = lmdb.open(path, map_size=2**30, sync=False, max_dbs=1)
            return LMDBPQStorage(env, clear)
    else:
        raise Exception('Wroing storage type: %s' % storage_type)
