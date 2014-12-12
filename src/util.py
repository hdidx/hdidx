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
