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

int PowConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto scope = op->scope();
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  VLOG(3) << "[MLU] Converting " + op_type + "...";

  // input
  auto input_var_name = op_info->Input("X").front();
  auto input = scope->FindVar(input_var_name)->GetMutable<lite::Tensor>();
  auto input_shape = input->dims().Vectorize();
  // output
  auto output_var_name = op_info->Output("Out").front();
  auto output = scope->FindVar(output_var_name)->GetMutable<lite::Tensor>();
  // attr
  auto factor = op_info->GetAttr<float>("factor");

  CHECK(graph->HasNode(input_var_name));
  auto input_tensor = graph->GetNode(input_var_name);
  auto output_tensor = graph->AddNode(output_var_name,
                                      output->dims().Vectorize(),
                                      CNML_TENSOR,
                                      CNML_NCHW,
                                      graph->FPType());


  cnmlBaseOp_t pow_op;
  CNML_CALL(cnmlCreatePowerOp(&pow_op,
                            input_tensor->mlu_tensor(),
                            output_tensor->mlu_tensor(),
                            factor));

  graph->FuseOp(pow_op);
  CNML_CALL(cnmlDestroyBaseOp(&pow_op));
  return SUCCESS;
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(pow,
                         kMLU,
                         paddle::lite::subgraph::mlu::PowConverter);
