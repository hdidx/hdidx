#!/usr/bin/env python
# coding: utf-8
"""
   File Name: setup.py
      Author: Wan Ji
      E-mail: wanji@live.com
  Created on: Tue Nov  4 08:55:15 2014 CST
 Description:
"""


try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

try:
    from pypandoc import convert

    def read_md(fpath):
        return convert(fpath, 'rst')
except ImportError:
    print "warning: pypandoc module not found, DONOT convert Markdown to RST"

    def read_md(fpath):
        with open(fpath, 'r') as fp:
            return fp.read()

import numpy
from Cython.Distutils import build_ext

setup(name='hdidx',
      version='0.2.8',
      # version='0.0.10052',
      author='WAN Ji',
      author_email='wanji@live.com',
      package_dir={'hdidx': 'hdidx'},
      packages=[
          'hdidx',
          'hdidx.indexer',
          'hdidx.encoder',
          'hdidx.storage',
      ],

      cmdclass={'build_ext': build_ext},
      ext_modules=[
          Extension(
              "hdidx._cext",
              sources=[
                  "cext/_cext.pyx",
                  "cext/cext.c",
              ],
              include_dirs=[numpy.get_include(), "cext"]),
          Extension(
              "hdidx._mih",
              sources=[
                  "cext/_mih.pyx",
                  "cext/mih.cpp",
              ],
              language="c++",
              include_dirs=[numpy.get_include(), "cext"])
      ],

      scripts=[
          'tests/test_indexer.py',
          'tests/test_cext.py',
          'tools/hdidx_build',
          'tools/hdidx_index',
      ],
      url='http://wanji.me/hdidx',
      license='LICENSE.md',
      description='ANN Search in High-Dimensional Spaces',

      long_description=open("DESC.rst").read(),
      install_requires=[
          "numpy      >= 1.6.0",
          "scipy      >= 0.9.0",
          "bottleneck >= 0.8.0",
          "lmdb       >= 0.83",
      ],
      )
