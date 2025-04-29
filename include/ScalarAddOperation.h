#include "Operations.h"

#pragma once

class ScalarAddOperation : public BinaryOperation {
public:
    ScalarAddOperation(const std::shared_ptr<INode> lhs, const Tensor& rhs): BinaryOperation(lhs, rhs) {}

    Tensor evaluate() const override {
        Tensor lhs_tensor = lhs_->evaluate();
        // Element-wise addition of lhs tensor and rhs tensor
        //assert(lhs_tensor.shape() == rhs_.shape() && "Shapes must match for addition");
        if(lhs_tensor.shape() == rhs_.shape()) throw std::invalid_argument("Shapes must match for addition.");
        return lhs_tensor + rhs_;
    }
};