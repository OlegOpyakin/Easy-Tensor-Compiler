#include "Operations.h"

#pragma once


class SoftmaxOperation : public UnaryOperation {
public:
    SoftmaxOperation(const std::shared_ptr<INode> arg): UnaryOperation(arg) {}
    SoftmaxOperation(const Tensor& tensor): UnaryOperation(tensor) {}

    Tensor evaluate() const override {
        Tensor input;
        if(is_node_) input = arg_->evaluate();
        else input = tensor_;
        
        std::vector<size_t> output_shape = input.shape();
        Tensor output(output_shape);
        
        size_t batch_size = input.shape()[0];
        size_t channels = input.shape()[1];
        size_t height = input.shape()[2];
        size_t width = input.shape()[3];
        
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                std::vector<float> exp_values(width * height);
                float exp_sum = 0.0f;

                for (size_t h = 0; h < height; ++h) {
                    for (size_t w = 0; w < width; ++w) {
                        exp_values[w + h * width] = std::exp(input.at(b, c, h, w));
                        exp_sum += exp_values[w + h * width];
                    }
                }

                for (size_t h = 0; h < height; ++h) {
                    for (size_t w = 0; w < width; ++w) {
                        output.at(b, c, h, w) = exp_values[w + h * width] / exp_sum;
                    }
                }
            }
        }
        
        return output;
    }
};
