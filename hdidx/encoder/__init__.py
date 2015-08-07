#!/usr/bin/env python
# coding: utf-8

"""
   File Name: encoder
      Author: Wan Ji
      E-mail: wanji@live.com
  Created on: Fri Jul 31 20:18:30 2015 CST
"""
DESCRIPTION = """
"""

import cPickle as pickle


class Encoder(object):
    """
    Encoder maps original data to hash codes.
    """
    def __init__(self):
        self.ERR_INSTAN = "Instance of `Encoder` is not allowed!"
        self.ERR_UNIMPL = "Unimplemented method!"
        pass

    def __del__(self):
        pass

    def build(self, vals=None, labels=None):
        """
        Build the encoder based on given training data.
        """
        raise Exception(self.ERR_INSTAN)

    def load(self, path):
        """
        Load encoder information from file.
        """
        with open(path, 'rb') as pklf:
            self.ecdat = pickle.load(pklf)

    def save(self, path):
        """
        Save the encoder information.
        """
        with open(path, 'wb') as pklf:
            pickle.dump(self.ecdat, pklf, protocol=2)

    def encode(self, vals):
        """
        Map `vals` to hash codes.
        """
        raise Exception(self.ERR_INSTAN)

from pq import PQEncoder, IVFPQEncoder
from sh import SHEncoder
