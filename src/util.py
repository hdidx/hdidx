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

"""
KMeans
"""
# from sklearn.cluster import KMeans
# from yael import ynumpy
import cv2

# distance
from distance import distFunc
import bottleneck

# profiling
import time


"""
Different options for kmeans.
1. kmeans_yael
2. kmeans_cv2
3. kmeans_sklearn
"""

"""
def kmeans_yael(vs, ks, niter):
    return ynumpy.kmeans(vs, ks, niter=niter, verbose=True)


def kmeans_sklearn(vs, ks, niter):
    kmeans = KMeans(n_clusters=ks, max_iter=niter)
    kmeans.fit(vs)
    return kmeans.cluster_centers_
"""


def kmeans_cv2(vs, ks, niter):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                niter, 0.01)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(
        vs, ks, criteria, 1, flags)
    return centers


kmeans = kmeans_cv2


# finding nearest neighbor

def pq_kmeans_assign(centroids, query):
    dist = distFunc['euclidean'](centroids, query)
    return dist.argmin(1)


def pq_knn(dist, topk):
    ids = bottleneck.argpartsort(dist, topk)[:topk]
    ids = ids[dist[ids].argsort()]
    return ids


# Profiling

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

    def str_overall(self, fmt="%s: %.4fs"):
        """
        Return the overall time costs for each code snippet as string.
        """

        return ";\t".join([fmt % (name, rec.time)
                           for name, rec in self.records.iteritems()])

    def str_average(self, fmt="%s: %.4f"):
        """
        Return the average time costs for each code snippet as string.
        """
        return ";\t".join([fmt % (name, rec.average())
                           for name, rec in self.records.iteritems()])

    def reset(self):
        """
        Reset the time costs and counters.
        """
        self.records = {}
        self.name_stack = []
        self.cur_record = None
