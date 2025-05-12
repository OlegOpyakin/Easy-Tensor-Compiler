#include "Operations.h"
#include "FastMatMul.h"
#include <iterator>

#pragma once


class MatMulOperation : public BinaryOperation {
public:
    MatMulOperation(const std::shared_ptr<INode> lhs, const std::shared_ptr<INode> rhs): BinaryOperation(lhs, rhs) {}
    MatMulOperation(const std::shared_ptr<INode> lhs, const Tensor& rhs_tensor): BinaryOperation(lhs, rhs_tensor) {}
    MatMulOperation(const Tensor& lhs_tensor, const std::shared_ptr<INode> rhs): BinaryOperation(lhs_tensor, rhs) {}
    MatMulOperation(const Tensor& lhs_tensor, const Tensor& rhs_tensor): BinaryOperation(lhs_tensor, rhs_tensor) {}

    Tensor evaluate() const override {
        Tensor lhs_tensor;
        Tensor rhs_tensor;
        
        if(lhs_is_node_) lhs_tensor = lhs_->evaluate();
        else lhs_tensor = lhs_tensor_;
        if(rhs_is_node_) rhs_tensor = rhs_->evaluate();
        else rhs_tensor = rhs_tensor_;

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
        if(lhs_batch_size != rhs_batch_size) throw std::invalid_argument("Batch size must be same for matrix multiplication.");
        if(lhs_channels != rhs_channels) throw std::invalid_argument("Channels size must be same for matrix multiplication.");
        if(lhs_width != rhs_height) throw std::invalid_argument("Incompatible dimensions for matrix multiplication.");
        
        // Create output tensor with shape [m, n]
        std::vector<size_t> result_shape = {lhs_batch_size, lhs_channels, lhs_height, rhs_width};
        std::vector<float> empty_data;

        Tensor result(result_shape, empty_data);
        for(size_t i = 0; i < result.size(); ++i) result.GetData()[i] = 0;
        
        // Old method
        /*
        for(size_t b = 0; b < lhs_batch_size; ++b)
            for(size_t c = 0; c < lhs_channels; ++c)
                for(size_t i = 0; i < lhs_height; ++i)
                    for(size_t j = 0; j < rhs_width; ++j)
                        for(size_t k = 0; k < lhs_width; ++k)
                            result.at(b, c, i, j) += lhs_tensor.at(b, c, i, k) * rhs_tensor.at(b, c, k, j);
        */

        // New method

        std::vector<float> tmp_left;
        std::vector<float> tmp_right;
        std::vector<float> tmp_result;
        size_t lhs_begin_copy_iter = 0;
        size_t lhs_end_copy_iter = 0;
        size_t rhs_begin_copy_iter = 0;
        size_t rhs_end_copy_iter = 0;

        for(size_t b = 0; b < lhs_batch_size; ++b)
            for(size_t c = 0; c < lhs_channels; ++c){
                
                lhs_begin_copy_iter = lhs_tensor.index(b, c, 0, 0);
                lhs_end_copy_iter = lhs_tensor.index(b, c, lhs_tensor.shape()[2] - 1, lhs_tensor.shape()[3] - 1);
                rhs_begin_copy_iter = rhs_tensor.index(b, c, 0, 0);
                rhs_end_copy_iter = rhs_tensor.index(b, c, rhs_tensor.shape()[2] - 1, rhs_tensor.shape()[3] - 1);
                
                //std::cout << "\n\nlhs_begin_copy_iter = " << lhs_begin_copy_iter << "\nlhs_end_copy_iter = " << lhs_end_copy_iter << "\n\n";
                
                for(size_t i = lhs_begin_copy_iter; i <= lhs_end_copy_iter; ++i)
                    tmp_left.push_back(lhs_tensor.GetData()[i]);
                for(size_t i = rhs_begin_copy_iter; i <= rhs_end_copy_iter; ++i)
                    tmp_right.push_back(rhs_tensor.GetData()[i]);
                
                std::cout << "\ntmp_left: ";
                for(auto it: tmp_left) std::cout << it << "  ";
                std::cout << "\ntmp_right: ";
                for(auto it: tmp_right) std::cout << it << "  ";

                std::cout << "\nlhs_height = " << lhs_height << "\nrhs_width = " << rhs_width << "\nlhs_width = " << lhs_width;

                tmp_result = MatrixMultiplyNeon::MatrixMultiplyFast(tmp_left, tmp_right, lhs_height, rhs_width, lhs_width);

                std::cout << "\ntmp_result: ";
                for(auto it: tmp_result) std::cout << it << "  ";
                
                for(size_t h = 0; h < lhs_height; ++h)
                    for(size_t w = 0; w < rhs_width; ++w)
                        result.at(b, c, h, w) = tmp_result[h * rhs_width + w];

                tmp_left.clear();
                tmp_right.clear();
                tmp_result.clear();
            }

        return result;
    }
};
