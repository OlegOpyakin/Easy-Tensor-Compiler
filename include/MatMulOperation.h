#include "Operations.h"

#pragma once


class MatMulOperation : public BinaryOperation {
public:
    MatMulOperation(const std::shared_ptr<INode> lhs, const std::shared_ptr<INode> rhs): BinaryOperation(lhs, rhs) {}

    Tensor evaluate() const override {
        Tensor lhs_tensor = lhs_->evaluate();
        Tensor rhs_tensor = rhs_->evaluate();

        // NCHW: [batch, channels, height, width]
        size_t lhs_batch_size = lhs_tensor.shape()[0];
        size_t lhs_channels = lhs_tensor.shape()[1];
        size_t lhs_height = lhs_tensor.shape()[2];
        size_t lhs_width = lhs_tensor.shape()[3];

        size_t rhs_batch_size = rhs_tensor.shape()[0];
        size_t rhs_channels = rhs_tensor.shape()[1];
        size_t rhs_height = rhs_tensor.shape()[2];
        size_t rhs_width = rhs_tensor.shape()[3];

        
        
        // Check validity for matrix multiplication
        //assert(lhs_tensor.shape(1) == rhs_.shape(0) && "Incompatible dimensions for matrix multiplication");
        if(lhs_batch_size != rhs_batch_size) throw std::invalid_argument("Batch size must be same for matrix multiplication.");
        if(lhs_channels != rhs_channels) throw std::invalid_argument("Channels size must be same for matrix multiplication.");
        if(lhs_width != rhs_height) throw std::invalid_argument("Incompatible dimensions for matrix multiplication.");
        
        // Create output tensor with shape [m, n]
        std::vector<size_t> result_shape = {lhs_batch_size, lhs_channels, lhs_height, rhs_width};
        std::vector<float> empty_data;

        Tensor result(result_shape, empty_data);
        
        for(size_t b = 0; b < lhs_batch_size; ++b)
            for(size_t c = 0; b < lhs_batch_size; ++b){
                for(size_t i = 0; i < lhs_height; ++i)
                    for(size_t j = 0; j < rhs_width; ++j)
                        for(size_t k = 0; k < lhs_width; ++k)
                        {
                            result.at(b, c, i, j) += 
                            mult[i][j] += a[i][k] * b[k][j];
                        }
            }
                

        
        return result;
    }
};
