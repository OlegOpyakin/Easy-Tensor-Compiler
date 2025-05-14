#include "Operations.h"
#include "FastMatMul.h"
#include "Im2Col.h"
#include <iterator>

#pragma once


class ConvolOperation : public BinaryOperation {
public:
    ConvolOperation(const std::shared_ptr<INode> lhs, const std::shared_ptr<INode> rhs, size_t stride, size_t padding): 
        BinaryOperation(lhs, rhs), stride_(stride), padding_(padding) {}
    ConvolOperation(const std::shared_ptr<INode> lhs, const Tensor& rhs_tensor, size_t stride, size_t padding): 
        BinaryOperation(lhs, rhs_tensor), stride_(stride), padding_(padding) {}
    ConvolOperation(const Tensor& lhs_tensor, const std::shared_ptr<INode> rhs, size_t stride, size_t padding): 
        BinaryOperation(lhs_tensor, rhs), stride_(stride), padding_(padding) {}
    ConvolOperation(const Tensor& lhs_tensor, const Tensor& rhs_tensor, size_t stride, size_t padding): 
        BinaryOperation(lhs_tensor, rhs_tensor), stride_(stride), padding_(padding) {}

    Tensor evaluate() const override {
        Tensor lhs_tensor;
        Tensor rhs_tensor;
        
        if(lhs_is_node_) lhs_tensor = lhs_->evaluate();
        else lhs_tensor = lhs_tensor_;
        if(rhs_is_node_) rhs_tensor = rhs_->evaluate();
        else rhs_tensor = rhs_tensor_;

        // Extract dimensions
        const size_t batch_size = lhs_tensor.shape(0);
        const size_t in_channels = lhs_tensor.shape(1);
        const size_t in_height = lhs_tensor.shape(2);
        const size_t in_width = lhs_tensor.shape(3);
        
        const size_t kernel_out_channels = rhs_tensor.shape(0);
        const size_t kernel_in_channels = rhs_tensor.shape(1);
        const size_t kernel_height = rhs_tensor.shape(2);
        const size_t kernel_width = rhs_tensor.shape(3);
        
        if (in_channels != kernel_in_channels) {
            throw std::invalid_argument("Input channels must match kernel input channels");
        }
        
        const size_t stride = stride_;
        const size_t padding = padding_;
        
        const size_t out_height = (in_height + 2 * padding - kernel_height) / stride + 1;
        const size_t out_width = (in_width + 2 * padding - kernel_width) / stride + 1;
        
        Tensor out_tensor(batch_size, kernel_out_channels, out_height, out_width);
        
        for (size_t batch = 0; batch < batch_size; ++batch) {
            performConvolutionIm2Col(
                lhs_tensor, rhs_tensor, out_tensor, batch, 
                in_channels, in_height, in_width,
                kernel_out_channels, kernel_height, kernel_width,
                stride, padding
            );
        }
        
        return out_tensor;
    }
    
private:
    size_t stride_;
    size_t padding_;

    void performConvolutionIm2Col(
        const Tensor& input, const Tensor& weights, Tensor& output,
        size_t batch,
        size_t in_channels, size_t in_height, size_t in_width,
        size_t out_channels, size_t kernel_height, size_t kernel_width,
        size_t stride, size_t padding
    ) const {
        // Calculate output dimensions
        const size_t out_height = (in_height + 2 * padding - kernel_height) / stride + 1;
        const size_t out_width = (in_width + 2 * padding - kernel_width) / stride + 1;
        
        // Create im2col matrix
        std::vector<float> im2col_data(kernel_height * kernel_width * in_channels * out_height * out_width, 0.0f);
        
        // Extract input data for the current batch
        std::vector<float> input_data(in_channels * in_height * in_width);
        for (size_t c = 0; c < in_channels; ++c) {
            for (size_t h = 0; h < in_height; ++h) {
                for (size_t w = 0; w < in_width; ++w) {
                    input_data[c * in_height * in_width + h * in_width + w] = input.at(batch, c, h, w);
                }
            }
        }
        
        // Apply im2col transform
        Im2Col(
            input_data.data(), in_channels, in_height, in_width,
            kernel_height, kernel_width, padding, padding, stride, stride,
            im2col_data.data()
        );
        
        // Reshape weights to match im2col for matrix multiplication
        std::vector<float> weight_matrix(out_channels * kernel_height * kernel_width * in_channels);
        for (size_t oc = 0; oc < out_channels; ++oc) {
            for (size_t ic = 0; ic < in_channels; ++ic) {
                for (size_t kh = 0; kh < kernel_height; ++kh) {
                    for (size_t kw = 0; kw < kernel_width; ++kw) {
                        const size_t weight_idx = oc * (in_channels * kernel_height * kernel_width) + 
                                                 ic * (kernel_height * kernel_width) + 
                                                 kh * kernel_width + kw;
                        weight_matrix[weight_idx] = weights.at(oc, ic, kh, kw);
                    }
                }
            }
        }
        
        std::vector<float> output_data = MatrixMultiplyNeon::MatrixMultiplyFast(
            weight_matrix, im2col_data,
            out_channels, out_height * out_width, kernel_height * kernel_width * in_channels
        );
        
        // Reshape output to NCHW format
        for (size_t oc = 0; oc < out_channels; ++oc) {
            for (size_t oh = 0; oh < out_height; ++oh) {
                for (size_t ow = 0; ow < out_width; ++ow) {
                    const size_t out_idx = oc * out_height * out_width + oh * out_width + ow;
                    output.at(batch, oc, oh, ow) = output_data[out_idx];
                }
            }
        }
    }
};