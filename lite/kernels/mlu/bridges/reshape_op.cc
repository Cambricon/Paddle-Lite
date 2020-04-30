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

int ReshapeConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[MLU] Converting " + op_type + "...";

  auto x_var_name = op_info->Input("X").front();
  auto out_var_name = op_info->Output("Out").front();
  auto x = scope->FindVar(x_var_name)->GetMutable<Tensor>();
  auto output = scope->FindVar(out_var_name)->GetMutable<Tensor>();
  auto output_dims = output->dims().Vectorize();
  auto output_tensor = graph->AddNode(
      out_var_name, output_dims, CNML_TENSOR, CNML_NCHW, graph->FPType());

  // =============== DEBUG ====================
  VLOG(6) << "x_var_name: " << x_var_name;
  VLOG(6) << "out_var_name: " << out_var_name;
  VLOG(6) << "output dim: " << output->dims();
  VLOG(6) << "input dim: " << x->dims();
  // =============== DEBUG END =================

  auto input_tensor = graph->GetNode(x_var_name);
  cnmlBaseOp_t reshape_op;

  cnmlReshapeOpParam_t reshape_param{nullptr};
  int cnml_out_shape[4];
  CNML_CALL(cnmlGetTensorShape(output_tensor->mlu_tensor(), cnml_out_shape));
  CNML_CALL(cnmlCreateNdReshapeOpParam(&reshape_param, cnml_out_shape, 4));

  // Use cnmlCreatexxxOpForward to create op.
  CNML_CALL(cnmlCreateReshapeOp(&reshape_op,
                                reshape_param,
                                input_tensor->mlu_tensor(),
                                output_tensor->mlu_tensor()));

  // CNML_CALL(cnmlCreateReshapeOp_V2(
  //     &reshape_op,
  //     input_tensor->mlu_tensor(),
  //     output_tensor->mlu_tensor()));
  graph->FuseOp(reshape_op);
  return SUCCESS;
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(reshape,
                         kMLU,
                         paddle::lite::subgraph::mlu::ReshapeConverter);
REGISTER_SUBGRAPH_BRIDGE(reshape2,
                         kMLU,
                         paddle::lite::subgraph::mlu::ReshapeConverter);
