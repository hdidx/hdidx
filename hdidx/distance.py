#!/usr/bin/env python
# coding: utf-8

#########################################################################
#########################################################################

"""
   File Name: distance.py
      Author: Wan Ji
      E-mail: wanji@live.com
  Created on: Mon Jun 30 13:23:40 2014 CST
"""
DESCRIPTION = """
Computing distances

Common parameters:
    is_sparse:
        is the feature sparse
    is_trans:
        is `feat` transfered
"""

import logging

import numpy as np
import scipy as sp


def fast_euclidean(feat, query, featl2norm):
    """ Euclidean distance.
    Args:
        feat:       N x D feature matrix
        query:      1 x D feature vector
        featl2norm: 1 x N vector

    Returns:
        dist:       1 x N vector
    """
    return ((query ** 2).sum() + featl2norm) - 2 * query.dot(feat.T)


def euclidean(feat, query, featl2norm=None, qryl2norm=None):
    """ Euclidean distance.
    Args:
        feat:       N x D feature matrix
        query:      Q x D feature vector
        featl2norm: 1 x N vector
        qryl2norm:  Q x 1 vector

    Returns:
        dist:       1 x N vector
    """

    dotprod = query.dot(feat.T)
    # return dotprod
    if qryl2norm is None:
        qryl2norm = (query ** 2).sum(1).reshape(-1, 1)
    if featl2norm is None:
        featl2norm = (feat ** 2).sum(1).reshape(1, -1)

    return - 2 * dotprod + qryl2norm + featl2norm


def Euclidean(feat, query=None,
              is_sparse=False, is_trans=False):
    """ Euclidean distance.
    """
    if query is None:
        (N, D) = feat.shape
        dotprod = feat.dot(feat.T)
        featl2norm = sp.repeat(dotprod.diagonal().reshape(1, -1), N, 0)
        qryl2norm = featl2norm.T
    else:
        (nQ, D) = query.shape
        (N, D) = feat.shape
        dotprod = query.dot(feat.T)
        qryl2norm = \
            sp.repeat(np.multiply(query, query).sum(1).reshape(-1, 1), N,  1)
        featl2norm = \
            sp.repeat(np.multiply(feat, feat).sum(1).reshape(1, -1), nQ, 0)

    return qryl2norm + featl2norm - 2 * dotprod


def Euclidean_DML(feat, M, query=None,
                  is_sparse=False, is_trans=False):
    """ Euclidean distance with DML.
    """
    (N, D) = feat.shape
    dotprod = feat.dot(M).dot(feat.T)
    l2norm = sp.repeat(dotprod.diagonal().reshape(1, -1), N, 0)
    return l2norm + l2norm.T - 2 * dotprod


def Cosine(feat, query=None,
           is_sparse=False, is_trans=False):
    """ Cosine distance.
    """
    logging.info("Cosine")
    try:
        feat = feat / np.sqrt(np.sum(np.multiply(feat, feat), axis=1))\
            .reshape((feat.shape[0], 1))
    except ValueError as e:
        logging.debug("%s (take feature as sparse matrix)" % e.message)
        feat2 = feat.copy()
        feat2.data **= 2
        rows_sums = np.array(feat2.sum(axis=1))[:, 0]
        feat.data /= rows_sums[feat2.nonzero()[0]]

    if query is None:
        query = feat
    else:
        try:
            query = query/np.sqrt(np.sum(np.multiply(query, query), axis=1))\
                .reshape((query.shape[0], 1))
        except ValueError as e:
            logging.debug("%s (take query as sparse matrix)" % e.message)
            query2 = query.copy()
            query2.data **= 2
            rows_sums = np.array(query2.sum(axis=1))[:, 0]
            query.data /= rows_sums[query2.nonzero()[0]]

    return DotProduct(feat, query)


def Cosine_DML(feat, M, query=None,
               is_sparse=False, is_trans=False):
    """ Cosine distance with DML.
    """
    try:
        feat = feat/np.sqrt(np.sum(np.multiply(feat, feat), axis=1))\
            .reshape((feat.shape[0], 1))
    except ValueError as e:
        logging.debug("%s (take feature as sparse matrix)" % e.message)
        feat2 = feat.copy()
        feat2.data **= 2
        rows_sums = np.array(feat2.sum(axis=1))[:, 0]
        feat.data /= rows_sums[feat2.nonzero()[0]]

    if query is None:
        query = feat
    else:
        try:
            query = query/np.sqrt(np.sum(np.multiply(query, query), axis=1))\
                .reshape((query.shape[0], 1))
        except ValueError as e:
            logging.debug("%s(take query as sparse matrix)" % e.message)
            query2 = query.copy()
            query2.data **= 2
            rows_sums = np.array(query2.sum(axis=1))[:, 0]
            query.data /= rows_sums[query2.nonzero()[0]]

    return DotProduct_DML(feat, M, query)


def DotProduct(feat, query=None,
               is_sparse=False, is_trans=False):
    """ DotProduct distance.
    """
    logging.debug("DotProduct")
    if query is None:
        query = feat
    return -query.dot(feat.T)


def DotProduct_DML(feat, M, query=None,
                   is_sparse=False, is_trans=False):
    """ DotProduct distance with DML.
    """
    if query is None:
        query = feat
    return -query.dot(M).dot(feat.T)


def DotProduct_DML_Diagonal(feat, M, query=None,
                            is_sparse=False, is_trans=False):
    """ DotProduct distance with DML.
    """
    if query is None:
        query = feat
    query.data = query.data * M[query.nonzero()[1]]
    return -query.dot(feat.T)


def Intersection(feat, query=None,
                 is_sparse=False, is_trans=False):
    """ Intersection distance.
    """
    raise Exception("Untested function")
    if query is None:
        query = feat
    qnum = query.shape[0]
    fnum = feat.shape[0]
    dist = np.zeros((qnum, fnum))

    for i in range(qnum):
        for j in range(fnum):
            dist[i, j] = sp.vstack((query[i, :], feat[j, :])).min(0).sum()
    return dist


def Intersection_DML(feat, M, query=None,
                     is_sparse=False, is_trans=False):
    """ Intersection distance with DML.
    """
    raise Exception("Untested function")
    if query is None:
        query = feat

    qnum = query.shape[0]
    fnum = feat.shape[0]
    dist = np.zeros((qnum, fnum))

    for i in range(qnum):
        query = query[i, :].dot(M)
        for j in range(fnum):
            dist[i, j] = sp.vstack((query, feat[j, :])).min(0).sum()
    return -query.dot(M).dot(feat.T)


distFunc = {
    "euclidean":    Euclidean,
    "dotproduct":   DotProduct,
    "intersection": Intersection,
    "cosine":       Cosine}


distFunc_DML = {
    "euclidean":    Euclidean_DML,
    "dotproduct":   DotProduct_DML,
    "intersection": Intersection_DML,
    "cosine":       Cosine_DML}
