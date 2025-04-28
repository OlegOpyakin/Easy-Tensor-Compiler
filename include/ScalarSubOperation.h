#include "Operations.h"

#pragma once

class ScalarSubOperation : public BinaryOperation {
public:
    ScalarSubOperation(const std::shared_ptr<INode> lhs, const Tensor& rhs): BinaryOperation(lhs, rhs) {}

    Tensor evaluate() const override {
        Tensor lhs_tensor = lhs_->evaluate();
        // Element-wise subtraction of rhs tensor from lhs tensor
        assert(lhs_tensor.shape() == rhs_.shape() && "Shapes must match for subtraction");
        return lhs_tensor - rhs_;
    }
};