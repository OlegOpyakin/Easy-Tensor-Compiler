#include <iostream>
#include <vector>
#include <cassert>
#include <stdexcept>
#include <memory>
#include <math.h>

#pragma once

//template<typename T>
class Tensor {
private:
    // NCHW: [batch, channels, height, width]
    std::vector<size_t> shape_;
    std::vector<float> data_;

public:
    // Default
    Tensor() : shape_({0, 0, 0, 0}), data_() {}

    // With shape parameters
    Tensor(size_t batch, size_t channels, size_t height, size_t width): shape_({batch, channels, height, width}) {
        data_.resize(batch * channels * height * width, 0.0f); // preserve memory
    }

    // With shape and initial
    Tensor(const std::vector<size_t>& shape, const std::vector<float>& data = {}): shape_(shape) {
        size_t size = 1;
        for (size_t dim : shape_) {
            size *= dim;
        }
        if (data.empty()) {
            data_.resize(size, 0.0f);
        } else {
            if(data.size() != size) throw std::length_error("Data size must match tensor size");
            data_ = data;
        }
    }

    // Copy ctor
    Tensor(const Tensor& other) : shape_(other.shape_), data_(other.data_) {}

    // Move ctor
    Tensor(Tensor&& other) noexcept: shape_(std::move(other.shape_)), data_(std::move(other.data_)) {}

    // Copy assignment
    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            shape_ = other.shape_;
            data_ = other.data_;
        }
        return *this;
    }

    // Move assignment
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            shape_ = std::move(other.shape_);
            data_ = std::move(other.data_);
        }
        return *this;
    }

    // Calculate index
    size_t index(size_t n, size_t c, size_t h, size_t w) const {
        if(n > shape_[0] || c > shape_[1] || h > shape_[2] || w > shape_[3]) throw std::out_of_range("Index out of range");
        return n * (shape_[1] * shape_[2] * shape_[3]) + c * (shape_[2] * shape_[3]) + h * shape_[3] + w;
    }

    // Get shape
    const std::vector<size_t>& shape() const {
        return shape_;
    }

    // Get dimension size
    size_t shape(size_t dim) const {
        if(dim > shape_.size()) throw std::out_of_range("Dimension index out of range");
        return shape_[dim];
    }

    // Get total number of elements
    size_t size() const {
        size_t total = 1;
        for (size_t dim : shape_) {
            total *= dim;
        }
        return total;
    }

    // Reshape tensor (total size must remain the same)
    void reshape(const std::vector<size_t>& new_shape) {
        size_t new_size = 1;
        for (size_t dim : new_shape) {
            new_size *= dim;
        }
        if(new_size != size()) throw std::length_error("New shape must have the same total size");
        shape_ = new_shape;
    }


    // Access element
    float& at(size_t n, size_t c, size_t h, size_t w) {
        return data_[index(n, c, h, w)];
    }

    // Access element const
    const float& at(size_t n, size_t c, size_t h, size_t w) const {
        return data_[index(n, c, h, w)];
    }

    // Access element flat
    float& at(size_t idx) {
        if(idx > data_.size()) throw std::out_of_range("Index out of range");
        return data_[idx];
    }

    // Access element flat const
    const float& at(size_t idx) const {
        if(idx > data_.size()) throw std::out_of_range("Index out of range");
        return data_[idx];
    }

    // Get all tensor to compare
    std::vector<float> GetData() { return data_; }

    // Tensor addition
    Tensor& operator+=(const Tensor& other) {
        if(shape_ != other.shape_) throw std::length_error("Tensors must have the same shape");
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] += other.data_[i];
        }
        return *this;
    }
    
    friend Tensor operator+(const Tensor& lhs, const Tensor& rhs) {
        Tensor result = lhs;
        result += rhs;
        return result;
    }

    // Tensor subtraction
    Tensor& operator-=(const Tensor& other) {
        if(shape_ != other.shape_) throw std::length_error("Tensors must have the same shape");
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] -= other.data_[i];
        }
        return *this;
    }

    friend Tensor operator-(const Tensor& lhs, const Tensor& rhs) {
        Tensor result = lhs;
        result -= rhs;
        return result;
    }

    // Scalar multiplication
    Tensor& operator*=(float scalar) {
        for (auto& val : data_) {
            val *= scalar;
        }
        return *this;
    }
   
    // Element-wise multiplication
    friend Tensor elementwise_mul(const Tensor& lhs, const Tensor& rhs) {
        if(lhs.shape_ != rhs.shape_) throw std::length_error("Tensors must have the same shape");
        Tensor result = lhs;
        for (size_t i = 0; i < result.data_.size(); ++i) {
            result.data_[i] *= rhs.data_[i];
        }
        return result;
    }

    

    // Print
    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
        os << "Tensor([" << tensor.shape_[0] << ", " << tensor.shape_[1] << ", " 
           << tensor.shape_[2] << ", " << tensor.shape_[3] << "], size=" 
           << tensor.data_.size() << ")";
        return os;
    }
};