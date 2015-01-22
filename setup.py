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
    read_md = lambda f: convert(f, 'rst')
except ImportError:
    print "warning: pypandoc module not found, DONOT convert Markdown to RST"
    read_md = lambda f: open(f, 'r').read()

import numpy
from Cython.Distutils import build_ext

setup(name='hdidx',
      version='0.0.11',
      author='WAN Ji',
      author_email='wanji@live.com',
      package_dir={'hdidx': 'src'},
      packages=['hdidx'],

      cmdclass={'build_ext': build_ext},
      ext_modules=[Extension("hdidx._cext",
                             sources=[
                                 "cext/_cext.pyx",
                                 "cext/cext.c",
                             ],
                             include_dirs=[numpy.get_include(), "cext"])],

      scripts=[
          'tests/test_indexer.py',
          'tests/test_cext.py',
      ],
      url='https://github.com/wanji/hdidx',
      # license='LICENSE.txt',
      description='ANN Search in High-Dimensional Spaces',

      long_description=read_md("DESC.md"),
      install_requires=[
          "numpy      >= 1.6.0",
          "scipy      >= 0.9.0",
          "bottleneck >= 0.8.0",
          "lmdb       >= 0.83",
      ],
      )
