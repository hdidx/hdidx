#!/usr/bin/env python
# coding: utf-8

#########################################################################
#########################################################################

"""
   File Name: test_cext.py
      Author: Wan Ji
      E-mail: wanji@live.com
  Created on: Wed Nov  5 16:41:24 2014 CST
"""
DESCRIPTION = """
"""

import unittest

# import time
# import logging
import numpy as np
# import ipdb

import hdidx._cext as cext
from hdidx.indexer.sh import SHIndexer as sh


class TestCExt(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_sumidxtab_core(self):
        nsq = 8
        ksub = 256
        cur_num = 10

        for iround in range(10):
            raw_D = np.random.random((nsq, ksub))
            raw_blk = np.random.random_integers(0, ksub-1, (cur_num, nsq))
            D = np.require(raw_D, np.float32, "C")
            blk = np.require(raw_blk, np.uint8, "C")

            self.assertLessEqual(np.abs(raw_D - D).sum(),  1e-4)
            self.assertEqual(np.abs(raw_blk - blk).sum(),  0)

            py_res = self.sumidxtab_core(D, blk)
            c_res = cext.sumidxtab_core(D, blk)
            self.assertLessEqual(np.abs(py_res - c_res).sum(),  1e-4)

    def sumidxtab_core(cls, D, blk):
        return np.require([sum([D[j, blk[i, j]] for j in range(D.shape[0])])
                           for i in range(blk.shape[0])], np.float32)

    def test_hamming_core(self):
        Q = (np.random.random((100, 4)) * 255).astype(np.uint8)
        D = (np.random.random((1000, 4)) * 255).astype(np.uint8)
        D1 = cext.hamming(Q, D)
        D2 = sh.hammingDist(Q, D)
        D3 = sh.hammingDist2(Q, D)

        self.assertEqual(self.ndarray_diff(D1, D2), 0)
        self.assertEqual(self.ndarray_diff(D1, D3), 0)

    def hamming_core(cls, D, blk):
        return np.require([sum([D[j, blk[i, j]] for j in range(D.shape[0])])
                           for i in range(blk.shape[0])], np.float32)

    def ndarray_diff(cls, A, B):
        return np.abs(A - B).max()


if __name__ == '__main__':
    unittest.main()
