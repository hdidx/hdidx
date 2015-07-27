#!/usr/bin/env python
# coding: utf-8

#########################################################################
#########################################################################

"""
   File Name: indexer.py
      Author: Wan Ji
      E-mail: wanji@live.com
  Created on: Tue Nov  4 09:07:38 2014 CST
"""
DESCRIPTION = """
"""

import cPickle as pickle


class Indexer(object):
    class IdxData(object):
        pass

    def __init__(self):
        self.ERR_INSTAN = "Instance of `Indexer` is not allowed!"
        self.ERR_UNIMPL = "Unimplemented method!"
        pass

    def __del__(self):
        pass

    def build(self, vals=None, labels=None):
        """
        Build the indexer based on given training data
        """
        raise Exception(self.ERR_INSTAN)

    def load(self, path):
        """
        Load indexer information from file
        """
        with open(path, 'rb') as pklf:
            self.idxdat = pickle.load(pklf)

    def save(self, path):
        """
        Save the information related to the indexer itself
        """
        with open(path, 'wb') as pklf:
            pickle.dump(self.idxdat, pklf, protocol=2)

    def set_storage(self, storage_type='mem', storage_parm=None):
        """
        Set up the backend storage engine
        """
        raise Exception(self.ERR_INSTAN)

    def add(self, vals, keys):
        """
        Add one or more items to the indexer
        """
        raise Exception(self.ERR_INSTAN)

    def remove(self, keys):
        """
        Remove one or more items from the indexer
        """
        raise Exception(self.ERR_INSTAN)

    def search(self, queries, topk=None, thresh=None):
        """
        Search in the indexer for `k` nearest neighbors or
        neighbors in a distance of `thresh`
        """
        raise Exception(self.ERR_INSTAN)

from pq import PQIndexer, IVFPQIndexer
