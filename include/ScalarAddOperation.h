#include "Operations.h"

#pragma once

class ScalarAddOperation : public BinaryOperation {
public:
    ScalarAddOperation(const std::shared_ptr<INode> lhs, const std::shared_ptr<INode> rhs): BinaryOperation(lhs, rhs) {}
    ScalarAddOperation(const std::shared_ptr<INode> lhs, const Tensor& rhs_tensor): BinaryOperation(lhs, rhs_tensor) {}
    ScalarAddOperation(const Tensor& lhs_tensor, const std::shared_ptr<INode> rhs): BinaryOperation(lhs_tensor, rhs) {}
    ScalarAddOperation(const Tensor& lhs_tensor, const Tensor& rhs_tensor): BinaryOperation(lhs_tensor, rhs_tensor) {}

    Tensor evaluate() const override {
        Tensor lhs_tensor;
        Tensor rhs_tensor;
        
        if(lhs_is_node_) lhs_tensor = lhs_->evaluate();
        else lhs_tensor = lhs_tensor_;
        if(rhs_is_node_) rhs_tensor = rhs_->evaluate();
        else rhs_tensor = rhs_tensor_;
        
        if(lhs_tensor.shape() != rhs_tensor.shape()) throw std::invalid_argument("Shapes must match for addition.");
        return lhs_tensor + rhs_tensor;
    }
};