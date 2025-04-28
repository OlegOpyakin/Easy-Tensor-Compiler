#include "Operations.h"

#pragma once

class ReLUOperation : public UnaryOperation {
public:
    ReLUOperation(const std::shared_ptr<INode> arg): UnaryOperation(arg) {}

    Tensor evaluate() const override {
        Tensor input = arg_->evaluate();
        Tensor output(input.shape());
        
        // ReLU: max(0, x)
        for (size_t i = 0; i < input.size(); ++i) {
            output.at(i) = std::max(0.0f, input.at(i));
        }
        
        return output;
    }
};