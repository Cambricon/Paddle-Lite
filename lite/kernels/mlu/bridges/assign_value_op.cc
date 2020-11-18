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

int AssignValueConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto scope = op->scope();
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  VLOG(3) << "[MLU] Converting " + op_type + "...";

  // input
//   auto input_var_name = op_info->Input("X").front();
//   auto input = scope->FindVar(input_var_name)->GetMutable<lite::Tensor>();
  // output
  auto output_var_name = op_info->Output("Out").front();
  auto output = scope->FindVar(output_var_name)->GetMutable<lite::Tensor>();
  auto buffer_size = output->dims().production();

  auto output_tensor = graph->AddNode(output_var_name,
                                      output->dims().Vectorize(),
                                      CNML_TENSOR,
                                      CNML_NCHW,
                                      graph->FPType());

  std::vector<float> fp32_values;
  std::vector<int> int32_values;
  auto* assign_data = const_cast<float*>(output->data<float>()); //reinterpret_cast<const void*>(output->data<float>());
    //   reinterpret_cast<float*>(malloc(buffer_size * sizeof(float)));
  CHECK(assign_data != nullptr);
  fp32_values = op_info->GetAttr<std::vector<float>>("fp32_values");
  if (0 != fp32_values.size()) {
    for (int i = 0; i < fp32_values.size(); i++) {
      assign_data[i] = fp32_values[i];
    }
  } else {
    int32_values = op_info->GetAttr<std::vector<int>>("int32_values");
    CHECK_EQ(buffer_size, int32_values.size());
    for (int i = 0; i < int32_values.size(); i++) {
      assign_data[i] = float(int32_values[i]);
    }
  }

  
  graph->BindConstData(output_var_name, output);

//   cnmlBaseOp_t assign_value_op;

//   graph->FuseOp(assign_value_op);
//   CNML_CALL(cnmlDestroyBaseOp(&assign_value_op));
  return SUCCESS;
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(assign_value,
                         kMLU,
                         paddle::lite::subgraph::mlu::AssignValueConverter);
