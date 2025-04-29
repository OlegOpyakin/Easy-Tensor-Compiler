#include "Operations.h"

#pragma once

/*
class MatMulOperation : public BinaryOperation {
public:
    MatMulOperation(const std::shared_ptr<INode> lhs, const Tensor& rhs): BinaryOperation(lhs, rhs) {}

    Tensor evaluate() const override {
        Tensor lhs_tensor = lhs_->evaluate();

        // FIX
        size_t m = lhs_tensor.shape(0);  // Rows of lhs
        size_t k = lhs_tensor.shape(1);  // Cols of lhs / Rows of rhs
        size_t n = rhs_.shape(1);        // Cols of rhs

        // NCHW: [batch, channels, height, width]
        size_t lhs_batch_size = lhs_tensor.shape()[0];
        size_t lhs_channels = lhs_tensor.shape()[1];
        size_t lhs_height = lhs_tensor.shape()[2];
        size_t lhs_width = lhs_tensor.shape()[3];

        size_t rhs_batch_size = rhs_.shape()[0];
        size_t rhs_channels = rhs_.shape()[1];
        size_t rhs_height = rhs_.shape()[2];
        size_t rhs_width = rhs_.shape()[3];

        // result = 
        
        // Check validity for matrix multiplication
        //assert(lhs_tensor.shape(1) == rhs_.shape(0) && "Incompatible dimensions for matrix multiplication");
        if(lhs_tensor.shape(1) == rhs_.shape(0)) throw std::invalid_argument("Incompatible dimensions for matrix multiplication.");
        
        // Create output tensor with shape [m, n]
        std::vector<size_t> result_shape = {m, n, 1, 1};
        std::vector<float> empty_data;

        Tensor result(result_shape, empty_data);
        // Perform matrix multiplication
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (size_t p = 0; p < k; ++p) {
                    sum += lhs_tensor.at(i, p, 0, 0) * rhs_.at(p, j, 0, 0);
                }
                result.at(i, j, 0, 0) = sum;
            }
        }
        
        return result;
    }
};
*/