/*************************************************************************
  > File Name: common.h
  > Copyright (C) 2013 Wan Ji<wanji@live.com>
  > Created Time: Fri 02 Oct 2015 15:44:21 CST
  > Descriptions: 
 ************************************************************************/

#ifndef _COMMON_H_
#define _COMMON_H_

#ifdef _MSC_VER

// typedef __int8 int8_t;
// typedef unsigned __int8 uint8_t;
// typedef __int16 int16_t;
// typedef unsigned __int16 uint16_t;
// typedef __int32 int32_t;
// typedef unsigned __int32 uint32_t;
// typedef __int64 int64_t;
// typedef unsigned __int64 uint64_t;


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

