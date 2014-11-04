#!/usr/bin/env python
# coding: utf-8

#########################################################################
#########################################################################

"""
   File Name: storage.py
      Author: Wan Ji
      E-mail: wanji@live.com
  Created on: Tue Nov  4 11:25:03 2014 CST
"""
DESCRIPTION = """
"""


class Storage(object):
    pass


class RedisStorage(Storage):
    pass


STORAGE_DIC = {
    'redis': RedisStorage,
}
