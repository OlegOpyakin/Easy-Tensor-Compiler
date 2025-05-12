#include "Operations.h"
#include "FastMatMul.h"
#include <iterator>

#pragma once


class ConvolOperation : public BinaryOperation {
private:
    Tensor pad_tensor(const Tensor& input, size_t padding) {
        if (padding == 0) return input;

        size_t N = input.shape()[0];
        size_t C = input.shape()[1];
        size_t H = input.shape()[2];
        size_t W = input.shape()[3];

        size_t H_pad = H + 2 * padding;
        size_t W_pad = W + 2 * padding;

        Tensor output({N, C, H_pad, W_pad});

        for (size_t n = 0; n < N; ++n) {
            for (size_t c = 0; c < C; ++c) {
                for (size_t h = 0; h < H; ++h) {
                    for (size_t w = 0; w < W; ++w) {
                        output.at(n, c, h + padding, w + padding) = input.at(n, c, h, w);
                    }
                }
            }
        }
        return output;
    }

    Tensor im2col(const Tensor& input, size_t kernel_size, size_t stride) {
        size_t N = input.shape()[0];
        size_t C = input.shape()[1];
        size_t H = input.shape()[2];
        size_t W = input.shape()[3];

        size_t H_out = (H - kernel_size) / stride + 1;
        size_t W_out = (W - kernel_size) / stride + 1;

        Tensor output({C * kernel_size * kernel_size, N * H_out * W_out});

        for (size_t n = 0; n < N; ++n) {
            for (size_t h_out = 0; h_out < H_out; ++h_out) {
                for (size_t w_out = 0; w_out < W_out; ++w_out) {
                    for (size_t c = 0; c < C; ++c) {
                        for (size_t kh = 0; kh < kernel_size; ++kh) {
                            for (size_t kw = 0; kw < kernel_size; ++kw) {
                                size_t h_in = h_out * stride + kh;
                                size_t w_in = w_out * stride + kw;

                                size_t col_row = c * kernel_size * kernel_size + kh * kernel_size + kw;
                                size_t col_col = n * H_out * W_out + h_out * W_out + w_out;

                                output({col_row, col_col}) = input({n, c, h_in, w_in});
                            }
                        }
                    }
                }
            }
        }

        return output;
    }

public:
    ConvolOperation(const std::shared_ptr<INode> lhs, const std::shared_ptr<INode> rhs): BinaryOperation(lhs, rhs) {}
    ConvolOperation(const std::shared_ptr<INode> lhs, const Tensor& rhs_tensor): BinaryOperation(lhs, rhs_tensor) {}
    ConvolOperation(const Tensor& lhs_tensor, const std::shared_ptr<INode> rhs): BinaryOperation(lhs_tensor, rhs) {}
    ConvolOperation(const Tensor& lhs_tensor, const Tensor& rhs_tensor): BinaryOperation(lhs_tensor, rhs_tensor) {}

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
        if(lhs_batch_size != rhs_batch_size) throw std::invalid_argument("Batch size must be same for convolution.");
        
        // Create output tensor with shape [m, n]
        std::vector<size_t> result_shape = {lhs_batch_size, lhs_channels * rhs_channels, lhs_height, rhs_width};
        std::vector<float> empty_data;

        Tensor result(result_shape, empty_data);
        for(size_t i = 0; i < result.size(); ++i) result.GetData()[i] = 0;
        
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
                

                for(size_t i = lhs_begin_copy_iter; i <= lhs_end_copy_iter; ++i)
                    tmp_left.push_back(lhs_tensor.GetData()[i]);
                for(size_t i = rhs_begin_copy_iter; i <= rhs_end_copy_iter; ++i)
                    tmp_right.push_back(rhs_tensor.GetData()[i]);
                
                tmp_result = MatrixMultiplyNeon::MatrixMultiplyFast(tmp_left, tmp_right, lhs_height, rhs_width, lhs_width);
                
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