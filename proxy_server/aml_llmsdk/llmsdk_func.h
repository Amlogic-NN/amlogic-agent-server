#ifndef __LLMSDK_FUNC_H__
#define __LLMSDK_FUNC_H__

#include <stdint.h>
#include "llmsdk_ext.h"

#ifdef __cplusplus
extern "C" {
#endif

int load_llmsdk_func(const char* lib_path);
void unload_llmsdk_func(void);

#ifdef __cplusplus
}
#endif

#endif