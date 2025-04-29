#include "Operations.h"

#pragma once

class ScalarMulOperation : public BinaryOperation {
public:
    ScalarMulOperation(const std::shared_ptr<INode> lhs, const Tensor& rhs): BinaryOperation(lhs, rhs) {}

    Tensor evaluate() const override {
        Tensor lhs_tensor = lhs_->evaluate();
        //assert(lhs_tensor.shape() == rhs_.shape() && "Shapes must match for element-wise multiplication");
        if(lhs_tensor.shape() == rhs_.shape()) throw std::invalid_argument("Shapes must match for element-wise multiplication.");
        return elementwise_mul(lhs_tensor, rhs_);
    }
};