#!/usr/bin/env python
# coding: utf-8

"""
   File Name: eval_indexer.py
      Author: Wan Ji
      E-mail: wanji@live.com
  Created on: Tue Jul 28 13:31:06 2015 CST
"""
DESCRIPTION = """
Evaluation the performance of HDIdx.
"""

import os
import argparse
import logging
import time
import tempfile

import numpy as np
from scipy.io import loadmat, savemat

from hdidx import indexer
from hdidx import util


def runcmd(cmd):
    """ Run command.
    """

    logging.info("%s" % cmd)
    return
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
    parser.add_argument("--nbits", type=int, nargs='+', default=[64],
                        help="number of bits")
    parser.add_argument("--topk", type=int, default=100,
                        help="retrieval `topk` nearest neighbors")
    parser.add_argument("--coarsek", type=int, default=1024,
                        help="size of the coarse codebook for IVFPQ")
    parser.add_argument("--eval_idx", type=int, default=-1,
                        help="index of v_indexer_param, -1 means all")
    parser.add_argument("--log", type=str, default="INFO",
                        help="log level")

    return parser.parse_args()


class Dataset(object):
    def __init__(self, data_path):
        data_dic = loadmat(data_path)
        for key, val in data_dic.iteritems():
            if not key.startswith("__"):
                setattr(self, key, val.T)
        self.groundtruth = self.groundtruth[:, :1]
        # self.query = self.query[:1000, :]
        self.nlrn = self.learn.shape[0]
        self.nbae = self.base.shape[0]
        self.nqry = self.query.shape[0]
        self.name = data_path.split("/")[-1].split(".")[0]
        # del self.base
        # del self.learn


def compute_stats_recall(ids_gnd, ids_qry, k):
    nquery = ids_qry.shape[0]
    n_pos = float(ids_gnd.shape[1])
    hit_map = np.zeros(ids_qry.shape, np.int)
    for i in xrange(nquery):
        hit_map[i, :] = [1 if ret_id in ids_gnd[i, :] else 0
                         for ret_id in ids_qry[i, :]]

    v_recall = []
    for i in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000,
              10000, 20000, 50000, 100000, 200000, 500000, 1000000]:
        if i > k:
            break
        hit_cnt = hit_map[:, :i].sum()
        r_at_i = float(hit_cnt) / n_pos / nquery
        logging.info('recall@%3d = %.4f' % (i, r_at_i))
        v_recall.append((i, r_at_i))
    return v_recall


def compute_stats_prec(ids_gnd, ids_qry, k):
    nquery = ids_qry.shape[0]
    hit_map = np.zeros(ids_qry.shape, np.int)
    for i in xrange(nquery):
        hit_map[i, :] = [1 if ret_id in ids_gnd[i, :] else 0
                         for ret_id in ids_qry[i, :]]

    v_prec = []
    for i in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000,
              10000, 20000, 50000, 100000, 200000, 500000, 1000000]:
        if i > k:
            break
        hit_cnt = hit_map[:, :i].sum()
        p_at_i = float(hit_cnt) / i / nquery
        logging.info('prec@%3d = %.4f' % (i, p_at_i))
        v_prec.append((i, p_at_i))
    return v_prec


compute_stats = compute_stats_prec
compute_stats = compute_stats_recall


BLOCK_SIZE = 1000


def save_result(rslt_path, ids, dis):
    nret = ids.shape[1]
    if nret > BLOCK_SIZE:
        if not os.path.exists(rslt_path):
            os.makedirs(rslt_path)
        for i in xrange(0, nret, BLOCK_SIZE):
            savemat(os.path.join(rslt_path, "%d.mat" % i),
                    {'ids': ids[:, i:i+BLOCK_SIZE],
                     'dis': dis[:, i:i+BLOCK_SIZE]})
    else:
        savemat(rslt_path, {'ids': ids, 'dis': dis})


def load_result(rslt_path):
    if os.path.isdir(rslt_path):
        v_ids = []
        v_dis = []
        files = sorted([rslt_file for rslt_file in os.listdir(rslt_path)
                        if rslt_file.endswith(".mat")],
                       key=lambda x: int(x.split(".")[0]))
        for rslt_file in files:
            rslt = loadmat(os.path.join(rslt_path, rslt_file))
            v_ids.append(rslt['ids'])
            v_dis.append(rslt['dis'])
        return np.hstack(v_ids), np.hstack(v_dis)
    else:
        rslt = loadmat(rslt_path)
        ids = rslt['ids']
        dis = rslt['dis']
    return ids, dis


def eval_indexer(data, indexer_param, dsname, topk):
    CurIndexer = indexer_param['indexer']
    build_param = indexer_param['build_param']
    search_param = indexer_param.get('search_param', {})
    index_prefix = indexer_param['index_prefix']
    info_path = index_prefix + ".info"
    lmdb_path = index_prefix + ".idx"
    rslt_path = index_prefix + "-top%d.mat" % topk

    logging.info("Start evaluating `%s`:" % CurIndexer.__name__)
    nbits = build_param['nbits'] if 'nbits' in build_param \
        else build_param['nsubq'] * 8
    logging.info("\tnumber of bits: %d" % nbits)

    runcmd("sh watch_mem.sh %d %s.mem &" % (os.getpid(), index_prefix))
    idx = CurIndexer()
    if os.path.exists(info_path):
        idx.load(info_path)
    else:
        build_param['vals'] = data.learn
        idx.build(build_param)
        idx.save(info_path)

    do_add = not os.path.exists(lmdb_path)

    runcmd("sleep 5")
    idx.set_storage('lmdb', {
        'path': lmdb_path,
        'clear': do_add,
    })
    if do_add:
        idx.add(data.base)

    if False and os.path.exists(rslt_path):
        logging.info("Loading saved retrieval results ...")
        ids, dis = load_result(rslt_path)
        logging.info("\tDone!")
        toc = 0
    else:
        logging.info("Searching ...")
        tic = time.time()
        ids, dis = idx.search(data.query, topk=topk, **search_param)
        toc = time.time() - tic
        save_result(rslt_path, ids, dis)
        logging.info("\tDone! (%.3fs)" % toc)
    runcmd("kill -9 `ps -ef | grep %d | grep watch_mem.sh | awk '{print $2}'`"
           % os.getpid())
    return compute_stats(data.groundtruth, ids, topk), toc


def main(args):
    """ Main entry.
    """
    exp_dir = args.exp_dir if args.exp_dir \
        else tempfile.mkdtemp(prefix="hdidx-eval-")
    logging.info("saving experimental results to: %s" % exp_dir)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    report = os.path.join(exp_dir, "report.txt")

    if os.path.exists(report):
        logging.warning("report file already exists")

    with open(report, "a") as rptf:
        rptf.write("*" * 64 + "\n")
        rptf.write("* %s\n" % time.asctime())
        rptf.write("*" * 64 + "\n")

    data = Dataset(args.dataset)
    logging.info("learn/base/query: %d/%d/%d" %
                 (data.nlrn, data.nbae, data.nqry))

    for nbits in args.nbits:
        if nbits % 8 != 0:
            raise util.HDIdxException("`nbits` must be multiple of 8")
        nsubq = nbits / 8
        v_indexer_param = [
            {
                'indexer': indexer.SHIndexer,
                'build_param': {
                    'nbits': nbits,
                },
                'index_prefix': '%s/%s_%s_nbits%d' % (
                    exp_dir, data.name, 'sh', nbits),
            },
            {
                'indexer': indexer.MIHIndexer,
                'build_param': {
                    'nbits': nbits,
                },
                'index_prefix': '%s/%s_%s_nbits%d' % (
                    exp_dir, data.name, 'mih', nbits),
            },
            {
                'indexer': indexer.PQIndexer,
                'build_param': {
                    'nsubq': nsubq,
                    'nsubqbits': 8,
                },
                'index_prefix': '%s/%s_%s_nsubq%d' % (
                    exp_dir, data.name, 'pq', nsubq),
            },
            {
                'indexer': indexer.IVFPQIndexer,
                'build_param': {
                    'nsubq': nsubq,
                    'nsubqbits': 8,
                    'coarsek': args.coarsek,
                },
                'search_param': {
                    'nn_coa': 10,
                },
                'index_prefix': '%s/%s_%s_nsubq%d_coarsek%d' % (
                    exp_dir, data.name, 'ivfpq', nsubq, args.coarsek),
            },
        ]

        if args.eval_idx >= 0:
            v_indexer_param = [v_indexer_param[args.eval_idx]]

        for indexer_param in v_indexer_param:
            v_recall, time_cost = eval_indexer(data, indexer_param,
                                               data.name, args.topk)
            with open(report, "a") as rptf:
                rptf.write("=" * 64 + "\n")
                # rptf.write(str(indexer_param) + "\n")
                rptf.write("index_%s-nbits_%s\n" % (
                    indexer_param['indexer'].__name__, nbits))
                rptf.write("-" * 64 + "\n")
                for recall in v_recall:
                    rptf.write("recall@%-8d%.4f\n" % (recall[0], recall[1]))
                rptf.write("time cost (ms): %.3f\n" %
                           (time_cost * 1000 / data.nqry))
    logging.info("All done! You can check the report in: %s" % report)

if __name__ == '__main__':
    args = getargs()
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: " + args.log)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
                        level=numeric_level)
    main(args)
