#include "Operations.h"

#pragma once

class ScalarSubOperation : public BinaryOperation {
public:
    ScalarSubOperation(const std::shared_ptr<INode> lhs, const std::shared_ptr<INode> rhs): BinaryOperation(lhs, rhs) {}

    Tensor evaluate() const override {
        Tensor lhs_tensor = lhs_->evaluate();
        Tensor rhs_tensor = rhs_->evaluate();

        if(lhs_tensor.shape() != rhs_tensor.shape()) throw std::invalid_argument("Shapes must match for subtraction.");
        return lhs_tensor - rhs_tensor;
    }
};