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

#include "lite/kernels/mlu/bridges/graph.h"
#include "lite/kernels/mlu/bridges/utility.h"
#include "lite/kernels/npu/bridges/registry.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

int DensityPriorBoxConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[MLU] Converting " + op_type + "...";

  auto input_name = op_info->Input("Input").front();
  auto image_name = op_info->Input("Image").front();
  auto boxes_name = op_info->Output("Boxes").front();
  auto variances_name = op_info->Output("Variances").front();

  auto input_var = scope->FindVar(input_name)->GetMutable<Tensor>();
  auto image_var = scope->FindVar(image_name)->GetMutable<Tensor>();
  auto boxes_var = scope->FindVar(boxes_name)->GetMutable<Tensor>();
  auto variances_var = scope->FindVar(variances_name)->GetMutable<Tensor>();

  auto input_dims = input_var->dims();
  auto image_dims = image_var->dims();
  auto boxes_dims = boxes_var->dims();
  auto variances_dims = variances_var->dims();

  auto clip = op_info->GetAttr<bool>("clip");
  auto fixed_sizes = op_info->GetAttr<std::vector<float>>("fixed_sizes");
  auto fixed_ratios = op_info->GetAttr<std::vector<float>>("fixed_ratios");
  auto variances_ = op_info->GetAttr<std::vector<float>>("variances");
  auto density_sizes = op_info->GetAttr<std::vector<int>>("density_sizes");
  auto offset = op_info->GetAttr<float>("offset");
  auto step_w = op_info->GetAttr<float>("step_w");
  auto step_h = op_info->GetAttr<float>("step_h");

  auto feat_tensor = graph->GetNode(input_name);
  auto image_tensor = graph->GetNode(image_name);
  auto boxes_tensor = graph->AddNode(boxes_name,
                                     boxes_dims.Vectorize(),
                                     CNML_TENSOR,
                                     CNML_NCHW,
                                     graph->FPType());
  auto variances_tensor = graph->AddNode(variances_name,
                                         variances_dims.Vectorize(),
                                         CNML_TENSOR,
                                         CNML_NCHW,
                                         graph->FPType());

  bool float32_precision = false;
  if (graph->FPType() == CNML_DATA_FLOAT32) {
    float32_precision = true;
  }

  // ==================== DEBUG ==================

  VLOG(6) << "input_name: " << input_name;
  VLOG(6) << "image_name: " << image_name;
  VLOG(6) << "boxes_name: " << boxes_name;
  VLOG(6) << "variances_name: " << variances_name;
  VLOG(6) << "input_dims : " << input_dims;
  VLOG(6) << "image_dims : " << image_dims;
  VLOG(6) << "boxes_dims : " << boxes_dims;
  VLOG(6) << "variances_dims : " << variances_dims;
  VLOG(6) << "clip : " << clip;
  VLOG(6) << "fixed_sizes : ";
  for (auto tmp : fixed_sizes) {
    VLOG(6) << tmp;
  }

  VLOG(6) << "fixed_ratios : ";
  for (auto tmp : fixed_ratios) {
    VLOG(6) << tmp;
  }
  VLOG(6) << "variances_ : ";
  for (auto tmp : variances_) {
    VLOG(6) << tmp;
  }
  VLOG(6) << "density_sizes : ";
  for (auto tmp : density_sizes) {
    VLOG(6) << tmp;
  }
  VLOG(6) << "offset : " << offset;
  VLOG(6) << "clip : " << clip;

  int feat_width = input_dims[3];
  int feat_height = input_dims[2];
  int image_width = image_dims[3];
  int image_height = image_dims[2];
  // ==================== DEBUG END ==================
  cnmlPluginDensityPriorBoxOpParam_t op_param;
  cnmlCreatePluginDensityPriorBoxOpParam(&op_param,
                                         feat_width,
                                         feat_height,
                                         image_width,
                                         image_height,
                                         variances_.data(),
                                         variances_.size(),
                                         density_sizes.data(),
                                         density_sizes.size(),
                                         fixed_sizes.data(),
                                         fixed_sizes.size(),
                                         fixed_ratios.data(),
                                         fixed_ratios.size(),
                                         clip,
                                         step_w,
                                         step_h,
                                         offset,
                                         float32_precision);

  cnmlTensor_t input_tensors[2];
  input_tensors[0] = feat_tensor->mlu_tensor();
  input_tensors[1] = image_tensor->mlu_tensor();
  cnmlTensor_t output_tensors[2];
  output_tensors[0] = boxes_tensor->mlu_tensor();
  output_tensors[1] = variances_tensor->mlu_tensor();
  cnmlBaseOp_t density_prior_box_op;
  CNML_CALL(cnmlCreatePluginDensityPriorBoxOp(
      &density_prior_box_op, op_param, input_tensors, output_tensors));
  graph->FuseOp(density_prior_box_op);
  return SUCCESS;
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(density_prior_box,
                         kMLU,
                         paddle::lite::subgraph::mlu::DensityPriorBoxConverter);
