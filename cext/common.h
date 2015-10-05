/*************************************************************************
  > File Name: common.h
  > Copyright (C) 2013 Wan Ji<wanji@live.com>
  > Created Time: Fri 02 Oct 2015 15:44:21 CST
  > Descriptions: 
 ************************************************************************/

#ifndef _COMMON_H_
#define _COMMON_H_

#ifdef _WIN32
typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int   uint32_t;
typedef unsigned long  uint64_t;

typedef char  int8_t;
typedef short int16_t;
typedef int   int32_t;
typedef long  int64_t;
#else
#include <stdint.h>
#endif

#endif

