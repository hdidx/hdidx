#!/usr/bin/env python
# coding: utf-8

"""
   File Name: eval_nearpy.py
      Author: Wan Ji
      E-mail: wanji@live.com
  Created on: Mon Aug 10 21:40:32 2015 CST
"""
DESCRIPTION = """
Evaluation the performance of RandomBinaryProjections from NearPy.
"""

import os
import argparse
import logging
import time
import itertools

import numpy as np

from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from nearpy.filters import NearestFilter, UniqueFilter

from hdidx.util import tic, toc
from eval_indexer import Dataset, compute_stats


def runcmd(cmd):
    """ Run command.
    """

    logging.info("%s" % cmd)
    os.system(cmd)


def getargs():
    """ Parse program arguments.
    """

    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('dataset', type=str,
                        help='path of the dataset')
    parser.add_argument('--exp_dir', type=str,
                        help='directory for saving experimental results')
    parser.add_argument("--nbits", type=int, nargs='+', default=[32, 16, 8],
                        help="number of bits per hash tables")
    parser.add_argument("--ntbls", type=int, nargs='+', default=[2, 4, 8],
                        help="number of hash tables")
    parser.add_argument("--topk", type=int, default=100,
                        help="retrieval `topk` nearest neighbors")
    parser.add_argument("--log", type=str, default="INFO",
                        help="log level")

    return parser.parse_args()


def main(args):
    """ Main entry.
    """

    data = Dataset(args.dataset)
    num, dim = data.base.shape

    # We are looking for the ten closest neighbours
    nearest = NearestFilter(args.topk)
    # We want unique candidates
    unique = UniqueFilter()

    # Create engines for all configurations
    for nbit, ntbl in itertools.product(args.nbits, args.ntbls):
        logging.info("Creating Engine ...")
        lshashes = [RandomBinaryProjections('rbp%d' % i, nbit)
                    for i in xrange(ntbl)]

        # Create engine with this configuration
        engine = Engine(dim, lshashes=lshashes,
                        vector_filters=[unique, nearest])
        logging.info("\tDone!")

        logging.info("Adding items ...")
        for i in xrange(num):
            engine.store_vector(data.base[i, :], i)
            if i % 100000 == 0:
                logging.info("\t%d/%d" % (i, data.nbae))
        logging.info("\tDone!")

        ids = np.zeros((data.nqry, args.topk), np.int)
        logging.info("Searching ...")
        tic()
        for i in xrange(data.nqry):
            reti = [y for x, y, z in
                    np.array(engine.neighbours(data.query[i]))]
            ids[i, :len(reti)] = reti
            if i % 100 == 0:
                logging.info("\t%d/%d" % (i, data.nqry))
        time_costs = toc()
        logging.info("\tDone!")

        report = os.path.join(args.exp_dir, "report.txt")
        with open(report, "a") as rptf:
            rptf.write("*" * 64 + "\n")
            rptf.write("* %s\n" % time.asctime())
            rptf.write("*" * 64 + "\n")

        r_at_k = compute_stats(data.groundtruth, ids, args.topk)[-1][-1]

        with open(report, "a") as rptf:
            rptf.write("=" * 64 + "\n")
            rptf.write("index_%s-nbit_%d-ntbl_%d\n" % ("NearPy", nbit, ntbl))
            rptf.write("-" * 64 + "\n")
            rptf.write("recall@%-8d%.4f\n" % (args.topk, r_at_k))
            rptf.write("time cost (ms): %.3f\n" %
                       (time_costs * 1000 / data.nqry))


if __name__ == '__main__':
    args = getargs()
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: " + args.log)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
                        level=numeric_level)
    main(args)
