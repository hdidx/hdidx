#!/usr/bin/env python
# coding: utf-8

"""
   File Name: lmdb_storage.py
      Author: Wan Ji
      E-mail: wanji@live.com
  Created on: Sat Aug  8 16:11:07 2015 CST
"""
DESCRIPTION = """
"""

import cPickle as pickle
from struct import pack, unpack
import numpy as np

from hdidx.storage import MemStorage


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
        # pkey = pack(ktype, key)
        # print len(pkey), [ord(c) for c in pkey], unpack('i', pkey)[0]
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


class LMDBStorage(MemStorage):
    def __init__(self, env, clear, ivfidx=None):
        MemStorage.__init__(self)

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
                    # print len(key), [ord(c) for c in key], unpack('i', key)[0]
                    keys.append(unpack('i', key)[0])
                    codes.append(pickle.loads(code))
                self.keys = np.array(keys)
                self.codes = np.vstack(tuple(codes))

    def __del__(self):
        self.db.close()

    def add(self, codes, keys):
        num_new_items = MemStorage.add(self, codes, keys)
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
