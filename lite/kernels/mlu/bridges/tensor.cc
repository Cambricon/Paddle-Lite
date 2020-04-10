// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/mlu/bridges/tensor.h"
#include <glog/logging.h>
#include <algorithm>
#include <climits>
#include <string>
#include <vector>

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

MLUTensor::MLUTensor(const std::vector<int64_t>& shape,
                     cnmlTensorType_t tensor_type,
                     cnmlDataOrder_t data_order,
                     cnmlDataType_t mlu_dtype)
    : mlu_tensor_(nullptr), tensor_type_(tensor_type), mlu_ptr_(nullptr) {
  std::vector<int> int_shape;
  for (auto i : shape) {
    if (i <= INT_MAX) {
      int_shape.push_back(i);
    } else {
      LOG(FATAL) << "Shape size is beyond the limitation of MLUTensor!";
    }
  }
  remember(int_shape, tensor_type, mlu_dtype, data_order);
}

void MLUTensor::remember(const std::vector<int>& shape,
                         cnmlTensorType_t tensor_type,
                         cnmlDataType_t mlu_dtype,
                         cnmlDataOrder_t shape_order) {
  tensor_type_ = tensor_type;
  mlu_dtype_ = mlu_dtype;
  origin_shape_.assign(shape.begin(), shape.end());

  int size = 4;
  if (shape.size() > 4 || shape_order == CNML_ARRAY) {
    size = shape.size();
  }
  shape_.resize(size);
  if (shape.size() <= 4) {
    switch (shape_order) {
      case CNML_NCHW:
        shape_[0] = shape.size() > 0 ? shape[0] : 1;
        shape_[3] = shape.size() > 1 ? shape[1] : 1;
        shape_[1] = shape.size() > 2 ? shape[2] : 1;
        shape_[2] = shape.size() > 3 ? shape[3] : 1;
        break;
      case CNML_NCWH:
        shape_[0] = shape.size() > 0 ? shape[0] : 1;
        shape_[3] = shape.size() > 1 ? shape[1] : 1;
        shape_[2] = shape.size() > 3 ? shape[3] : 1;
        shape_[1] = shape.size() > 2 ? shape[2] : 1;
        break;
      case CNML_NHWC:
        shape_[0] = shape.size() > 0 ? shape[0] : 1;
        shape_[3] = shape.size() > 3 ? shape[3] : 1;
        shape_[1] = shape.size() > 1 ? shape[1] : 1;
        shape_[2] = shape.size() > 2 ? shape[2] : 1;
        break;
      case CNML_NHCW:
        shape_[0] = shape.size() > 0 ? shape[0] : 1;
        shape_[3] = shape.size() > 2 ? shape[2] : 1;
        shape_[1] = shape.size() > 1 ? shape[1] : 1;
        shape_[2] = shape.size() > 3 ? shape[3] : 1;
        break;
      case CNML_NWCH:
        shape_[0] = shape.size() > 0 ? shape[0] : 1;
        shape_[3] = shape.size() > 2 ? shape[2] : 1;
        shape_[1] = shape.size() > 3 ? shape[3] : 1;
        shape_[2] = shape.size() > 1 ? shape[1] : 1;
        break;
      case CNML_NWHC:
        shape_[0] = shape.size() > 0 ? shape[0] : 1;
        shape_[3] = shape.size() > 3 ? shape[3] : 1;
        shape_[1] = shape.size() > 2 ? shape[2] : 1;
        shape_[2] = shape.size() > 1 ? shape[1] : 1;
        break;
      case CNML_CNHW:
        shape_[0] = shape.size() > 1 ? shape[1] : 1;
        shape_[3] = shape.size() > 0 ? shape[0] : 1;
        shape_[1] = shape.size() > 2 ? shape[2] : 1;
        shape_[2] = shape.size() > 3 ? shape[3] : 1;
        break;
      case CNML_CNWH:
        shape_[0] = shape.size() > 1 ? shape[1] : 1;
        shape_[3] = shape.size() > 0 ? shape[0] : 1;
        shape_[1] = shape.size() > 3 ? shape[3] : 1;
        shape_[2] = shape.size() > 2 ? shape[2] : 1;
        break;
      case CNML_CHWN:
        shape_[0] = shape.size() > 3 ? shape[3] : 1;
        shape_[3] = shape.size() > 0 ? shape[0] : 1;
        shape_[1] = shape.size() > 1 ? shape[1] : 1;
        shape_[2] = shape.size() > 2 ? shape[2] : 1;
        break;
      case CNML_CHNW:
        shape_[0] = shape.size() > 2 ? shape[2] : 1;
        shape_[3] = shape.size() > 0 ? shape[0] : 1;
        shape_[1] = shape.size() > 1 ? shape[1] : 1;
        shape_[2] = shape.size() > 3 ? shape[3] : 1;
        break;
      case CNML_CWNH:
        shape_[0] = shape.size() > 2 ? shape[2] : 1;
        shape_[3] = shape.size() > 0 ? shape[0] : 1;
        shape_[1] = shape.size() > 3 ? shape[3] : 1;
        shape_[2] = shape.size() > 1 ? shape[1] : 1;
        break;
      case CNML_CWHN:
        shape_[0] = shape.size() > 3 ? shape[3] : 1;
        shape_[3] = shape.size() > 0 ? shape[0] : 1;
        shape_[1] = shape.size() > 2 ? shape[2] : 1;
        shape_[2] = shape.size() > 1 ? shape[1] : 1;
        break;
      case CNML_HNCW:
        shape_[0] = shape.size() > 1 ? shape[1] : 1;
        shape_[3] = shape.size() > 2 ? shape[2] : 1;
        shape_[1] = shape.size() > 0 ? shape[0] : 1;
        shape_[2] = shape.size() > 3 ? shape[3] : 1;
        break;
      case CNML_HNWC:
        shape_[0] = shape.size() > 1 ? shape[1] : 1;
        shape_[3] = shape.size() > 3 ? shape[3] : 1;
        shape_[1] = shape.size() > 0 ? shape[0] : 1;
        shape_[2] = shape.size() > 2 ? shape[2] : 1;
        break;
      case CNML_HCWN:
        shape_[0] = shape.size() > 3 ? shape[3] : 1;
        shape_[3] = shape.size() > 1 ? shape[1] : 1;
        shape_[1] = shape.size() > 0 ? shape[0] : 1;
        shape_[2] = shape.size() > 2 ? shape[2] : 1;
        break;
      case CNML_HCNW:
        shape_[0] = shape.size() > 2 ? shape[2] : 1;
        shape_[3] = shape.size() > 1 ? shape[1] : 1;
        shape_[1] = shape.size() > 0 ? shape[0] : 1;
        shape_[2] = shape.size() > 3 ? shape[3] : 1;
        break;
      case CNML_HWNC:
        shape_[0] = shape.size() > 2 ? shape[2] : 1;
        shape_[3] = shape.size() > 3 ? shape[3] : 1;
        shape_[1] = shape.size() > 0 ? shape[0] : 1;
        shape_[2] = shape.size() > 1 ? shape[1] : 1;
        break;
      case CNML_HWCN:
        shape_[0] = shape.size() > 3 ? shape[3] : 1;
        shape_[3] = shape.size() > 2 ? shape[2] : 1;
        shape_[1] = shape.size() > 0 ? shape[0] : 1;
        shape_[2] = shape.size() > 1 ? shape[1] : 1;
        break;
      case CNML_WNCH:
        shape_[0] = shape.size() > 1 ? shape[1] : 1;
        shape_[3] = shape.size() > 2 ? shape[2] : 1;
        shape_[1] = shape.size() > 3 ? shape[3] : 1;
        shape_[2] = shape.size() > 0 ? shape[0] : 1;
        break;
      case CNML_WNHC:
        shape_[0] = shape.size() > 1 ? shape[1] : 1;
        shape_[3] = shape.size() > 3 ? shape[3] : 1;
        shape_[1] = shape.size() > 2 ? shape[2] : 1;
        shape_[2] = shape.size() > 0 ? shape[0] : 1;
        break;
      case CNML_WCHN:
        shape_[0] = shape.size() > 3 ? shape[3] : 1;
        shape_[3] = shape.size() > 1 ? shape[1] : 1;
        shape_[1] = shape.size() > 2 ? shape[2] : 1;
        shape_[2] = shape.size() > 0 ? shape[0] : 1;
        break;
      case CNML_WCNH:
        shape_[0] = shape.size() > 2 ? shape[2] : 1;
        shape_[3] = shape.size() > 1 ? shape[1] : 1;
        shape_[1] = shape.size() > 3 ? shape[3] : 1;
        shape_[2] = shape.size() > 0 ? shape[0] : 1;
        break;
      case CNML_WHNC:
        shape_[0] = shape.size() > 2 ? shape[2] : 1;
        shape_[3] = shape.size() > 3 ? shape[3] : 1;
        shape_[1] = shape.size() > 1 ? shape[1] : 1;
        shape_[2] = shape.size() > 0 ? shape[0] : 1;
        break;
      case CNML_WHCN:
        shape_[0] = shape.size() > 3 ? shape[3] : 1;
        shape_[3] = shape.size() > 2 ? shape[2] : 1;
        shape_[1] = shape.size() > 1 ? shape[1] : 1;
        shape_[2] = shape.size() > 0 ? shape[0] : 1;
        break;
      case CNML_ARRAY:
        shape_ = shape;
        break;
      default:
        LOG(FATAL) << "Unsupported mluDataOrder! " << int(shape_order);
        break;
    }
  } else {
    switch (shape_order) {
      case CNML_NCDHW:
        shape_[0] = shape[0];
        shape_[4] = shape[1];
        shape_[1] = shape[2];
        shape_[2] = shape[3];
        shape_[3] = shape[4];
        break;
      case CNML_NDHWC:
        shape_[0] = shape[0];
        shape_[4] = shape[4];
        shape_[1] = shape[1];
        shape_[2] = shape[2];
        shape_[3] = shape[3];
        break;
      case CNML_DHWCN:
        shape_[0] = shape[4];
        shape_[4] = shape[3];
        shape_[1] = shape[0];
        shape_[2] = shape[1];
        shape_[3] = shape[2];
        break;
      case CNML_ARRAY:
        shape_ = shape;
        break;
      default:
        shape_[0] = shape[0];
        shape_[4] = shape[1];
        shape_[1] = shape[2];
        shape_[2] = shape[3];
        shape_[3] = shape[4];
        break;
    }
  }
  dim_ = shape_.size();
}

void MLUTensor::Create() {
  if (mlu_tensor_ == nullptr) {
    CNML_CALL(cnmlCreateTensor_V2(&mlu_tensor_, tensor_type_));
    std::vector<int> dim_shape(shape_);
    int* dim_strides = nullptr;
    CNML_CALL(cnmlSetTensorShape_V2(
        mlu_tensor_, dim_, dim_shape.data(), dim_strides));
    CNML_CALL(cnmlSetTensorDataType(mlu_tensor_, mlu_dtype_));
  }
}

cnmlTensor_t MLUTensor::mlu_tensor() {
  Create();
  return mlu_tensor_;
}

void MLUTensor::ToFile(std::string file_name) {
  if (mlu_ptr_) {
    VLOG(5) << "to dump mlu ptr: " << mlu_ptr_ << " to: " << file_name
            << std::endl;
    int count = 1;
    for (size_t i = 0; i < shape_.size(); i++) {
      count *= shape_[i];
    }
    VLOG(6) << " dump count: " << count << std::endl;
    VLOG(6) << " dump shape: " << std::endl;
    for (size_t i = 0; i < shape_.size(); i++) {
      VLOG(6) << shape_[i] << " ";
    }

    VLOG(6) << std::endl;

    std::vector<float> cpu_data_fp32(count);
    // fp16 to fp32
    if (mlu_dtype_ == CNML_DATA_FLOAT16) {
      VLOG(6) << " convert fp16 to fp32 " << std::endl;
      std::vector<uint16_t> cpu_data_fp16(count);
      cnrtMemcpy(cpu_data_fp16.data(),
                 mlu_ptr_,
                 count * sizeof(uint16_t),
                 CNRT_MEM_TRANS_DIR_DEV2HOST);
      for (size_t i = 0; i < count; i++) {
        cnrtConvertHalfToFloat(&(cpu_data_fp32[i]), cpu_data_fp16[i]);
      }
    } else {
      cnrtMemcpy(cpu_data_fp32.data(),
                 mlu_ptr_,
                 count * sizeof(float),
                 CNRT_MEM_TRANS_DIR_DEV2HOST);
    }

    // trans to nchw
    std::vector<float> cpu_data_trans(count);
    transpose(
        cpu_data_fp32.data(), cpu_data_trans.data(), shape_, {0, 3, 1, 2});

    // to file
    std::ofstream of;
    of.open(file_name, std::ios::out);
    for (size_t i = 0; i < count; i++) {
      of << cpu_data_trans[i] << std::endl;
    }
    of.close();
  } else {
    LOG(FATAL) << "mlu ptr is null ,can not dump mlu content to : " << file_name
               << std::endl;
  }
}

MLUTensor::~MLUTensor() {
  if (mlu_tensor_ != nullptr) {
    CNML_CALL(cnmlDestroyTensor(&mlu_tensor_));
    mlu_tensor_ = nullptr;
  }
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
