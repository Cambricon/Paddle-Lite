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

#include "lite/kernels/mlu/transpose_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace mlu {

void TransposeCompute::Run() {
  auto& mlu_context = this->ctx_->template As<MLUContext>();
  auto& exec_queue = mlu_context.exec_queue();
  this->Run(exec_queue);
}

void TransposeCompute::Run(const cnrtQueue_t& exec_queue) {
  auto& param = this->Param<param_t>();

  // auto* rois = param.ROIs;
  // auto rois_dims = rois->dims();
  // int rois_num = rois_dims[0];
  // if (rois_num == 0) {
  //   return;
  // }

  auto* in = param.x;
  auto* out = param.output;
  auto in_dims = in->dims();
  // int batch_size = in_dims[0];
  auto out_dims = out->dims();
  // auto* xshape = param.xshape;
  std::vector<int> axis = param.axis;
  int x0 = 1, x1 = 1, x2 = 1, x3 = 1, x4 = 1, x5 = 1, x6 = 1, x7 = 1;
  int p0 = 1, p1 = 1, p2 = 1, p3 = 1, p4 = 1, p5 = 1, p6 = 1, p7 = 1;
  int dims = 2, max_idx = 0, size_dt = sizeof(float),
      sum_num = in_dims.production();
  mluDataType_t data_type = MLU_DTYPE_FLOAT;
  mluTransposeStrategy_t strategy;
  if (axis.size() == 4) {
    // if (axis == {0, 2, 3, 1}) {
    if (axis[0] == 0 && axis[1] == 2 && axis[2] == 3 && axis[3] == 1) {
      x0 = in_dims[0];
      x1 = in_dims[1];
      x2 = in_dims[3] * in_dims[2];
      p0 = in_dims[0];
      p1 = in_dims[2] * in_dims[3];
      p2 = in_dims[1];
      strategy = TRANSPOSE_3D_021;
    } else if (axis[0] == 0 && axis[1] == 3 && axis[2] == 1 && axis[3] == 2) {
      // else if (axis == {0, 3, 1, 2}) {
      x0 = in_dims[0];
      x1 = in_dims[1] * in_dims[2];
      x2 = in_dims[3];
      p0 = in_dims[0];
      p1 = in_dims[3];
      p2 = in_dims[1] * in_dims[2];
      strategy = TRANSPOSE_3D_021;
    } else if (axis[0] == 0 && axis[1] == 1 && axis[2] == 2 && axis[3] == 3) {
      // else if (axis == {0, 1, 2, 3}) {
      x0 = in_dims[0];
      x1 = in_dims[1] * in_dims[2];
      x2 = in_dims[3];
      p0 = in_dims[0];
      p1 = in_dims[1] * in_dims[2];
      p2 = in_dims[3];
      strategy = TRANSPOSE_3D_021;
    }

  } else if (axis.size() == 3) {
    // else if (axis == {0, 2, 1}) {
    if (axis[0] == 0 && axis[1] == 2 && axis[2] == 1) {
      x0 = in_dims[0];
      x1 = in_dims[1];
      x2 = in_dims[2];
      p0 = in_dims[0];
      p1 = in_dims[2];
      p2 = in_dims[1];
      strategy = TRANSPOSE_3D_021;
    } else if (axis[0] == 0 && axis[1] == 1 && axis[2] == 2) {
      // else if (axis == {0, 1, 2}) {
      x0 = in_dims[0];
      x1 = in_dims[1];
      x2 = in_dims[2];
      p0 = in_dims[0];
      p1 = in_dims[1];
      p2 = in_dims[2];
      strategy = TRANSPOSE_3D_021;
    }

  } else {
    if (axis[0] == 1 && axis[1] == 0) {
      x0 = in_dims[0];
      x1 = in_dims[1];
      p0 = in_dims[1];
      p1 = in_dims[0];
      strategy = TRANSPOSE_2D;
    } else {
      //  else if (axis == {0, 1}) {
      x0 = in_dims[0];
      x1 = in_dims[1];
      p0 = in_dims[0];
      p1 = in_dims[1];
      strategy = TRANSPOSE_2D;
    }
  }
  // int pooled_height = param.pooled_height;
  // int pooled_width = param.pooled_width;
  // int sampling_ratio = param.sampling_ratio;

  // half spatial_scale_half;
  // cnrtConvertFloatToHalf(&spatial_scale_half, spatial_scale);

  auto* input_data = in->data<float>();
  auto* output_data = out->mutable_data<float>();

  // prepare kernel params
  cnrtKernelParamsBuffer_t params;
  cnrtGetKernelParamsBuffer(&params);
  cnrtKernelParamsBufferAddParam(params, &input_data, sizeof(input_data));
  cnrtKernelParamsBufferAddParam(params, &output_data, sizeof(output_data));

  cnrtKernelParamsBufferAddParam(params, &x0, sizeof(x0));
  cnrtKernelParamsBufferAddParam(params, &x1, sizeof(x1));
  cnrtKernelParamsBufferAddParam(params, &x2, sizeof(x2));
  cnrtKernelParamsBufferAddParam(params, &x3, sizeof(x3));
  cnrtKernelParamsBufferAddParam(params, &x4, sizeof(x4));
  cnrtKernelParamsBufferAddParam(params, &x5, sizeof(x5));
  cnrtKernelParamsBufferAddParam(params, &x6, sizeof(x6));
  cnrtKernelParamsBufferAddParam(params, &x7, sizeof(x7));
  cnrtKernelParamsBufferAddParam(params, &p0, sizeof(p0));
  cnrtKernelParamsBufferAddParam(params, &p1, sizeof(p1));
  cnrtKernelParamsBufferAddParam(params, &p2, sizeof(p2));
  cnrtKernelParamsBufferAddParam(params, &p3, sizeof(p3));
  cnrtKernelParamsBufferAddParam(params, &p4, sizeof(p4));
  cnrtKernelParamsBufferAddParam(params, &p5, sizeof(p5));
  cnrtKernelParamsBufferAddParam(params, &p6, sizeof(p6));
  cnrtKernelParamsBufferAddParam(params, &p7, sizeof(p7));
  cnrtKernelParamsBufferAddParam(params, &dims, sizeof(dims));
  cnrtKernelParamsBufferAddParam(params, &max_idx, sizeof(max_idx));
  cnrtKernelParamsBufferAddParam(params, &size_dt, sizeof(size_dt));
  cnrtKernelParamsBufferAddParam(params, &sum_num, sizeof(sum_num));
  cnrtKernelParamsBufferAddParam(params, &data_type, sizeof(data_type));
  cnrtKernelParamsBufferAddParam(params, &strategy, sizeof(strategy));

  cnrtDim3_t task_dims;
  task_dims.x = 4, task_dims.y = 1, task_dims.z = 1;
  cnrtFunctionType_t func_type = CNRT_FUNC_TYPE_UNION1;

  // invoke kernel and sync to compute on MLU
  CNRT_CALL(cnrtInvokeKernel_V2(reinterpret_cast<void**>(&MLUTransposeKernel),
                                task_dims,
                                params,
                                func_type,
                                exec_queue));
  CNRT_CALL(cnrtSyncQueue(exec_queue));

  // realease resource
  cnrtDestroyKernelParamsBuffer(params);
  // cnrtFree(input_mlu_data);
  // cnrtFree(rois_mlu_data);
  // cnrtFree(roi_batch_id_mlu_data);
  // cnrtFree(output_mlu_data);
}

}  // namespace mlu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(transpose,
                     kMLU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::mlu::TransposeCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();
