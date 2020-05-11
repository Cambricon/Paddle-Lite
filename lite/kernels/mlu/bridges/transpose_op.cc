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

int TransposeConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[MLU] Converting " + op_type + "...";

  // Get input vars and op attributes
  auto x_var_name = op_info->Input("X").front();
  auto x = scope->FindVar(x_var_name)->GetMutable<Tensor>();
  auto x_dims = x->dims().Vectorize();

  auto out_var_name = op_info->Output("Out").front();
  auto output = scope->FindVar(out_var_name)->GetMutable<Tensor>();
  auto output_dims = output->dims().Vectorize();

  auto axis = op_info->GetAttr<std::vector<int>>("axis");
  while (axis.size() < 4) {
    axis.push_back(axis.size());
  }

  CHECK(graph->HasNode(x_var_name));

  // ================== Trans1: NHWC => NCHW ===========================
  auto input_tensor = graph->GetNode(x_var_name);
  std::vector<int> nhwc_to_nchw_axis = {0, 3, 1, 2};
  auto trans1_out = graph->AddNode(
      x_var_name + ".trans.i", x_dims, CNML_TENSOR, CNML_NHWC, graph->FPType());
  cnmlBaseOp_t trans1_op{nullptr};
  cnmlNdTransposeOpParam_t trans1_param{nullptr};
  CNML_CALL(cnmlCreateNdTransposeOpParam(
      &trans1_param, nhwc_to_nchw_axis.data(), nhwc_to_nchw_axis.size()));
  CNML_CALL(cnmlCreateNdTransposeProOp(&trans1_op,
                                       input_tensor->mlu_tensor(),
                                       trans1_out->mlu_tensor(),
                                       trans1_param));
  // ======================== Trans1 End ==================================

  // ======================= Transpose op ===================================
  auto trans2_input = graph->AddNode(out_var_name + ".trans.o",
                                     output_dims,
                                     CNML_TENSOR,
                                     CNML_NHWC,
                                     graph->FPType());
  cnmlBaseOp_t transpose_op{nullptr};
  cnmlNdTransposeOpParam_t transpose_param{nullptr};
  CNML_CALL(
      cnmlCreateNdTransposeOpParam(&transpose_param, axis.data(), axis.size()));
  // Use cnmlCreatexxxOpForward to create op.
  CNML_CALL(cnmlCreateNdTransposeProOp(&transpose_op,
                                       trans1_out->mlu_tensor(),
                                       trans2_input->mlu_tensor(),
                                       transpose_param));
  // ======================= Transpose op End
  // ===================================

  // ================== Trans2: NCHW => NHWC ===============================
  std::vector<int> nchw_to_nhwc_axis = {0, 2, 3, 1};
  auto output_tensor = graph->AddNode(
      out_var_name, output_dims, CNML_TENSOR, CNML_NCHW, graph->FPType());
  cnmlBaseOp_t trans2_op{nullptr};
  cnmlNdTransposeOpParam_t trans2_param{nullptr};
  CNML_CALL(cnmlCreateNdTransposeOpParam(
      &trans2_param, nchw_to_nhwc_axis.data(), nchw_to_nhwc_axis.size()));
  CNML_CALL(cnmlCreateNdTransposeProOp(&trans2_op,
                                       trans2_input->mlu_tensor(),
                                       output_tensor->mlu_tensor(),
                                       trans2_param));
  // ======================== Trans2 End ==================================
  // =============== DEBUG LOG ======================
  VLOG(6) << "x_var_name :" << x_var_name;
  VLOG(6) << "x_dims :" << x->dims();
  VLOG(6) << "out_var_name :" << out_var_name;
  VLOG(6) << "output_dims :" << output->dims();
  VLOG(6) << "axis :";
  for (size_t i = 0; i < axis.size(); i++) {
    VLOG(6) << axis[i];
  }
  int tmp_shape[4];
  cnmlGetTensorShape(trans1_out->mlu_tensor(), tmp_shape);
  VLOG(6) << "trans1_out shape"
          << ": " << tmp_shape[0] << " " << tmp_shape[1] << " " << tmp_shape[2]
          << " " << tmp_shape[3];
  cnmlGetTensorShape(trans2_input->mlu_tensor(), tmp_shape);
  VLOG(6) << "trans2_input shape"
          << ": " << tmp_shape[0] << " " << tmp_shape[1] << " " << tmp_shape[2]
          << " " << tmp_shape[3];
  // =============== DEBUG END ======================

  graph->FuseOp(trans1_op);
  graph->FuseOp(transpose_op);
  graph->FuseOp(trans2_op);
  CNML_CALL(cnmlDestroyBaseOp(&trans1_op));
  CNML_CALL(cnmlDestroyBaseOp(&transpose_op));
  CNML_CALL(cnmlDestroyBaseOp(&trans2_op));
  VLOG(6) << "transpose convert finished";
  return SUCCESS;
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
REGISTER_SUBGRAPH_BRIDGE(transpose,
                         kMLU,
                         paddle::lite::subgraph::mlu::TransposeConverter);
REGISTER_SUBGRAPH_BRIDGE(transpose2,
                         kMLU,
                         paddle::lite::subgraph::mlu::TransposeConverter);
