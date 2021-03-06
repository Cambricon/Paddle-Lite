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

#include "lite/operators/while_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool WhileOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.sub_block);
  CHECK_OR_FALSE(param_.scope);
  CHECK_OR_FALSE(param_.cond);
  return true;
}

bool WhileOpLite::InferShapeImpl() const { return true; }

bool WhileOpLite::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  auto inputs = op_desc.Input("X");
  auto outs = op_desc.Output("Out");

  for (auto var : inputs) {
    // param_.x.push_back(scope->FindVar(var)->GetMutable<lite::Tensor>());
  }
  for (auto var : outs) {
    // param_.outs.push_back(scope->FindVar(var)->GetMutable<lite::Tensor>());
  }
  param_.sub_block = sub_block_;

  auto condition = op_desc.Input("Condition");
  param_.cond = scope->FindVar(condition[0])->GetMutable<lite::Tensor>();
  param_.scope = scope;

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(while, paddle::lite::operators::WhileOpLite);
