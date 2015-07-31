#!/usr/bin/env python
# coding: utf-8

"""
   File Name: sh.py
      Author: Wan Ji
      E-mail: wanji@live.com
  Created on: Fri Jul 31 20:26:32 2015 CST
"""
DESCRIPTION = """
"""

import logging

import numpy as np

from hdidx.encoder import Encoder
from hdidx.util import eigs


class SHEncoder(Encoder):
    def __init__(self):
        Encoder.__init__(self)

    def __del__(self):
        pass

    def build(self, pardic=None):
        # training data
        X = pardic['vals']
        # the number of subquantizers
        nbits = pardic['nbits']
        # the number of items in one block
        blksize = pardic.get('blksize', 16384)

        [Nsamples, Ndim] = X.shape

        # algo:
        # 1) PCA
        npca = min(nbits, Ndim)
        pc, l = eigs(np.cov(X.T), npca)
        X = X.dot(pc)   # no need to remove the mean

        # 2) fit uniform distribution
        eps = np.finfo(float).eps
        mn = np.percentile(X, 5)
        mx = np.percentile(X, 95)
        mn = X.min(0) - eps
        mx = X.max(0) + eps

        # 3) enumerate eigenfunctions
        R = mx - mn
        maxMode = np.ceil((nbits+1) * R / R.max())
        nModes = maxMode.sum() - maxMode.size + 1
        modes = np.ones((nModes, npca))
        m = 0
        for i in xrange(npca):
            modes[m+1:m+maxMode[i], i] = np.arange(1, maxMode[i]) + 1
            m = m + maxMode[i] - 1
        modes = modes - 1
        omega0 = np.pi / R
        omegas = modes * omega0.reshape(1, -1).repeat(nModes, 0)
        eigVal = -(omegas ** 2).sum(1)
        ii = (-eigVal).argsort()
        modes = modes[ii[1:nbits+1], :]

        """
        Initializing encoder data
        """
        ecdat = dict()
        ecdat['nbits'] = nbits
        ecdat['pc'] = pc
        ecdat['mn'] = mn
        ecdat['mx'] = mx
        ecdat['modes'] = modes
        ecdat['blksize'] = blksize
        self.ecdat = ecdat

    @staticmethod
    def compactbit(b):
        nSamples, nbits = b.shape
        nwords = (nbits + 7) / 8
        B = np.hstack([np.packbits(b[:, i*8:(i+1)*8][:, ::-1], 1)
                       for i in xrange(nwords)])
        residue = nbits % 8
        if residue != 0:
            B[:, -1] = np.right_shift(B[:, -1], 8 - residue)

        return B

    def encode(self, vals):
        X = vals
        if X.ndim == 1:
            X = X.reshape((1, -1))

        Nsamples, Ndim = X.shape
        nbits = self.ecdat['nbits']
        mn = self.ecdat['mn']
        mx = self.ecdat['mx']
        pc = self.ecdat['pc']
        modes = self.ecdat['modes']

        X = X.dot(pc)
        X = X - mn.reshape((1, -1))
        omega0 = 0.5 / (mx - mn)
        omegas = modes * omega0.reshape((1, -1))

        U = np.zeros((Nsamples, nbits))
        for i in range(nbits):
            omegai = omegas[i, :]
            ys = X * omegai + 0.25
            ys -= np.floor(ys)
            yi = np.sum(ys < 0.5, 1)
            U[:, i] = yi

        b = np.require(U % 2 == 0, dtype=np.int)
        B = self.compactbit(b)
        return B
