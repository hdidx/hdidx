#!/usr/bin/env python
# coding: utf-8

"""
   File Name: eval_annoy.py
      Author: Wan Ji
      E-mail: wanji@live.com
  Created on: Mon Aug 10 21:40:32 2015 CST
"""
DESCRIPTION = """
"""

import os
import argparse
import logging
import time

import numpy as np

from annoy import AnnoyIndex

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
    parser.add_argument("--ntrees", type=int, nargs='+', default=[16],
                        help="number of trees")
    parser.add_argument("--topk", type=int, default=100,
                        help="retrieval `topk` nearest neighbors")
    parser.add_argument("--log", type=str, default="INFO",
                        help="log level")

    return parser.parse_args()


def main(args):
    """ Main entry.
    """

    data = Dataset(args.dataset)
    f = data.base.shape[1]

    for ntrees in args.ntrees:
        t = AnnoyIndex(f)   # Length of item vector that will be indexed
        idxpath = os.path.join(args.exp_dir, 'sift_annoy_ntrees%d.idx' % ntrees)
        if not os.path.exists(idxpath):
            logging.info("Adding items ...")
            for i in xrange(data.nbae):
                t.add_item(i, data.base[i])
                if i % 100000 == 0:
                    logging.info("\t%d/%d" % (i, data.nbae))
            logging.info("\tDone!")
            logging.info("Building indexes ...")
            t.build(ntrees)
            logging.info("\tDone!")
            t.save(idxpath)
        else:
            logging.info("Loading indexes ...")
            t.load(idxpath)
            logging.info("\tDone!")

        ids = np.zeros((data.nqry, args.topk), np.int)
        logging.info("Searching ...")
        tic()
        for i in xrange(data.nqry):
            ids[i, :] = np.array(t.get_nns_by_vector(data.query[i], args.topk))
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
            rptf.write("index_%s-ntrees_%s\n" % ("Annoy", ntrees))
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
