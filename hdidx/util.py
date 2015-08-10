#!/usr/bin/env python
# coding: utf-8

#########################################################################
#########################################################################

"""
   File Name: util.py
      Author: Wan Ji
      E-mail: wanji@live.com
  Created on: Tue Nov  4 09:38:24 2014 CST
"""
DESCRIPTION = """
"""

import os
import logging
# distance
from distance import distFunc
import bottleneck
from scipy.io import loadmat
import numpy as np

# profiling
import time


DO_NORM = {
    "cosine": True,
    "euclidean": False,
}


class HDIdxException(Exception):
    """
    HDIdx Exception
    """


"""
Math
"""


def eigs(X, npca):
    l, pc = np.linalg.eig(X)
    idx = l.argsort()[::-1][:npca]
    return pc[:, idx], l[idx]


"""
KMeans
"""

try:
    import cv2

    def kmeans(vs, ks, niter):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    niter, 0.01)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels, centers = cv2.kmeans(
            vs, ks, criteria, 1, flags)
        return centers
except ImportError:
    logging.warn("Cannot find OpenCV, using `kmeans` from SciPy instead.")
    from scipy.cluster import vq

    def kmeans(vs, ks, niter):
        centers, labels = vq.kmeans2(vs, ks, niter)
        return centers


# finding nearest neighbor


def pq_kmeans_assign(centroids, query):
    dist = distFunc['euclidean'](centroids, query)
    return dist.argmin(1)


def pq_knn(dist, topk):
    ids = bottleneck.argpartsort(dist, topk)[:topk]
    ids = ids[dist[ids].argsort()]
    return ids


# Profiling
START_TIME = 0


def tic():
    global START_TIME
    START_TIME = time.time()


def toc():
    return time.time() - START_TIME


class Profiler(object):
    """ Profiling the running time of code snippet.
    """
    class Record(object):
        __slots__ = ["name", "time", "count", "t0"]

        def __init__(self, name):
            self.name = name
            self.reset()

        def reset(self):
            self.time = 0.0
            self.count = 0
            self.t0 = None

        def start(self):
            self.t0 = time.time()

        def end(self):
            self.time += (time.time() - self.t0)
            self.count += 1
            self.t0 = None

        def average(self):
            return self.time / self.count if self.count > 0 else 0

    __slots__ = ["records",
                 "cur_record",
                 "name_stack"]

    def __init__(self):
        self.reset()

    def start(self, name):
        """
        Start the timer.
        `name` is the description of the current code snippet.
        """
        if name not in self.records:
            self.records[name] = Profiler.Record(name)
        self.cur_record = self.records[name]
        self.name_stack.append(name)
        self.cur_record.start()

    def end(self, name=None):
        """
        Calculate the time costs of the current code snippet.
        """
        if name is not None and name != self.name_stack[-1]:
            raise Exception("name '%s' should be '%s'" %
                            (name, self.name_stack[-1]))
        self.cur_record.end()
        self.name_stack.pop()

    def sum_overall(self):
        """
        Return the sum of overall time costs for each code snippet.
        """
        return sum([rec.time for name, rec in self.records.iteritems()])

    def sum_average(self):
        """
        Return the sum of average time costs for each code snippet.
        """
        return sum([rec.average() for name, rec in self.records.iteritems()])

    def str_overall(self, fmt="%s: %.3fms"):
        """
        Return the overall time costs for each code snippet as string.
        """

        return ";\t".join([fmt % (name, rec.time * 1000)
                           for name, rec in self.records.iteritems()])

    def str_average(self, fmt="%s: %.3fms"):
        """
        Return the average time costs for each code snippet as string.
        """
        return ";\t".join([fmt % (name, rec.average() * 1000)
                           for name, rec in self.records.iteritems()])

    def reset(self):
        """
        Reset the time costs and counters.
        """
        self.records = {}
        self.name_stack = []
        self.cur_record = None


def normalize(feat, ln=2):
    if ln is 1:
        return feat / feat.sum(1).reshape(-1, 1)
    elif ln > 0:
        return feat / ((feat**ln).sum(1)**(1.0/ln)).reshape(-1, 1)
    else:
        raise Exception("Unsupported norm: %d" % ln)


def tokey(item):
    """
    Key function for sorting filenames
    """
    return int(item.split("_")[-1].split(".")[0])


class Reader(object):
    def __init__(self, featdir):
        self.v_fname = sorted(os.listdir(featdir), key=tokey)
        self.next_id = 0
        self.featdir = featdir

    def get_next(self):
        logging.info("Reader - load %d" % self.next_id)
        feat = loadmat(
            os.path.join(self.featdir, self.v_fname[self.next_id]))['feat']
        self.next_id += 1
        return feat
