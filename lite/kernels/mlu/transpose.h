// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.ddNod
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef KERNELS_TRANSPOSE_TRANSPOSE_H_
#define KERNELS_TRANSPOSE_TRANSPOSE_H_

#include <vector>
// #include "kernels/kernel.h"
// #include "cnnl.h"

#define BANG_TRANSPOSE_ALIGN_FP16 (64)
#define MEMCPY_NRAM2NRAM_ALIGN_FP16 (64)
#define BANG_TRANSPOSE_ALIGN_FP32 (32)
#define MEMCPY_NRAM2NRAM_ALIGN_FP32 (32)
#define BANG_TRANSPOSE_ALIGN_INT8 (128)
#define MEMCPY_NRAM2NRAM_ALIGN_INT8 (128)

#define BANG_TRANSPOSE_ALIGN_BYTE (128)
#define MEMCPY_NRAM2NRAM_ALIGN_BYTE (128)

#define NRAM_NUM_LIMIT_FP32 (30 * 1024)
#define NRAM_NUM_LIMIT_FP16 (2 * NRAM_NUM_LIMIT_FP32)
#define NRAM_BYTE_LIMIT (4 * NRAM_NUM_LIMIT_FP32)
#define NRAM_BYTE_LIMIT_ALIGNED (NRAM_BYTE_LIMIT + 4096)

#define TRANSPOSE_MAX_DIM (8)
#define TRANSPOSE_MAX_U1_JOB (4)
#define TRANSPOSE_MAX_BLOCK_JOB (4)

struct cnnlTransposeStruct {
  int dim;
  std::vector<int> permute;
};

typedef enum cnnlTransposeStrategy {
  TRANSPOSE_1D,
  TRANSPOSE_2D,
  TRANSPOSE_3D_021,
  TRANSPOSE_3D_102,
  TRANSPOSE_3D_210,
  TRANSPOSE_4D_0213,
  TRANSPOSE_4D_0321,
  TRANSPOSE_4D_1032,
  TRANSPOSE_4D_1302,
  TRANSPOSE_4D_1320,
  TRANSPOSE_4D_2031,
  TRANSPOSE_4D_2103,
  TRANSPOSE_4D_2130,
  TRANSPOSE_4D_3021,
  TRANSPOSE_4D_3102,
  TRANSPOSE_4D_3210,
  TRANSPOSE_COMMON,
} cnnlTransposeStrategy_t;

typedef enum cnnlDataType {
  CNNL_DTYPE_INT32,
  CNNL_DTYPE_FLOAT,
  CNNL_DTYPE_HALF,
  CNNL_DTYPE_INT16,
  CNNL_DTYPE_INT8,
  CNNL_DTYPE_BOOL,
  CNNL_DTYPE_INT31,
} cnnlDataType_t;

typedef enum cnnlStatus {
  CNNL_STATUS_SUCCESS,
  CNNL_STATUS_ALLOC_FAILED,
  CNNL_STATUS_BAD_PARAM,
} cnnlStatus_t;

typedef struct cnnlTransposeStruct *cnnlTransposeDescriptor_t;

struct cnnlContext {
  cnrtDev_t device;
  cnrtQueue_t queue;
  int32_t cluster_num;
  int32_t core_num_per_cluster;
};

typedef struct cnnlContext *cnnlHandle_t;

int32_t getNumOfUnionCapability(cnnlHandle_t handle) {
  // CHECK(handle != NULL);
  return handle->cluster_num;
}

struct cnnlTensorStruct {
  cnnlTensorStruct()
      : dim(0),
        dtype(CNNL_DTYPE_FLOAT),  // layout(CNNL_LAYOUT_ARRAY),
        position(0),
        scale(1.0),
        offset(0.0) {}
  ~cnnlTensorStruct() {}
  cnnlStatus_t tensorDimN(size_t &dim);
  cnnlStatus_t tensorDimC(size_t &dim);
  cnnlStatus_t tensorDimH(size_t &dim);
  cnnlStatus_t tensorDimW(size_t &dim);
  cnnlStatus_t tensorElementsNumber(size_t &elements) {
    uint64_t elements_counter = 1;
    // for (auto dimension : dims) {
    for (int i = 0; i < dims.size(); i++) {
      int dimension = dims[i];
      elements_counter *= dimension;
    }
    elements = elements_counter;
    return CNNL_STATUS_SUCCESS;
  };
  cnnlStatus_t tensorSize(size_t &size);
  int dim;
  std::vector<int> dims;
  cnnlDataType_t dtype;
  // cnnlTensorLayout_t layout;
  int position;
  float scale;
  float offset;
};

typedef struct cnnlTensorStruct *cnnlTensorDescriptor_t;

uint64_t cnnlGetTensorElementNum(cnnlTensorDescriptor_t desc) {
  uint64_t tensor_num = 1;
  cnnlStatus_t return_status = desc->tensorElementsNumber(tensor_num);
  CHECK(return_status == CNNL_STATUS_SUCCESS);
  return tensor_num;
}

void MLUTransposeKernel(void *x,
                        void *y,
                        const int x0,
                        const int x1,
                        const int x2,
                        const int x3,
                        const int x4,
                        const int x5,
                        const int x6,
                        const int x7,
                        const int p0,
                        const int p1,
                        const int p2,
                        const int p3,
                        const int p4,
                        const int p5,
                        const int p6,
                        const int p7,
                        const int dims,
                        const int max_idx,
                        const int size_dt,
                        const int sum_num,
                        cnnlDataType_t type,
                        cnnlTransposeStrategy_t strategy);

#endif  // KERNELS_TRANSPOSE_TRANSPOSE_H_
