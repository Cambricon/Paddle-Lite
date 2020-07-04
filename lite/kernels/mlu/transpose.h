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

#ifndef LITE_KERNELS_MLU_TRANSPOSE_H_
#define LITE_KERNELS_MLU_TRANSPOSE_H_

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

struct mluTransposeStruct {
  int dim;
  std::vector<int> permute;
};

typedef enum mluTransposeStrategy {
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
} mluTransposeStrategy_t;

typedef enum mluDataType {
  MLU_DTYPE_INT32,
  MLU_DTYPE_FLOAT,
  MLU_DTYPE_HALF,
  MLU_DTYPE_INT16,
  MLU_DTYPE_INT8,
  MLU_DTYPE_BOOL,
  MLU_DTYPE_INT31,
} mluDataType_t;

typedef enum mluStatus {
  MLU_STATUS_SUCCESS,
  MLU_STATUS_ALLOC_FAILED,
  MLU_STATUS_BAD_PARAM,
} mluStatus_t;

typedef struct mluTransposeStruct *mluTransposeDescriptor_t;

struct mluContext {
  cnrtDev_t device;
  cnrtQueue_t queue;
  int32_t cluster_num;
  int32_t core_num_per_cluster;
};

typedef struct mluContext *mluHandle_t;

int32_t getNumOfUnionCapability(mluHandle_t handle) {
  // CHECK(handle != NULL);
  return handle->cluster_num;
}

struct mluTensorStruct {
  mluTensorStruct()
      : dim(0),
        dtype(MLU_DTYPE_FLOAT),  // layout(MLU_LAYOUT_ARRAY),
        position(0),
        scale(1.0),
        offset(0.0) {}
  ~mluTensorStruct() {}
  // mluStatus_t tensorElementsNumber(size_t &elements) {
  //   uint64_t elements_counter = 1;
  //   // for (auto dimension : dims) {
  //   for (int i = 0; i < dims.size(); i++) {
  //     int dimension = dims[i];
  //     elements_counter *= dimension;
  //   }
  //   elements = elements_counter;
  //   return MLU_STATUS_SUCCESS;
  // }
  mluStatus_t tensorSize(const size_t &size);
  int dim;
  std::vector<int> dims;
  mluDataType_t dtype;
  // mluTensorLayout_t layout;
  int position;
  float scale;
  float offset;
};

typedef struct mluTensorStruct *mluTensorDescriptor_t;

// uint64_t mluGetTensorElementNum(mluTensorDescriptor_t desc) {
//   uint64_t tensor_num = 1;
//   mluStatus_t return_status = desc->tensorElementsNumber(tensor_num);
//   CHECK(return_status == MLU_STATUS_SUCCESS);
//   return tensor_num;
// }

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
                        mluDataType_t type,
                        mluTransposeStrategy_t strategy);

#endif  // LITE_KERNELS_MLU_TRANSPOSE_H_
