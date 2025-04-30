#include "Tensor.h"
#include <concepts>
#include <type_traits>
#include <iostream>

#pragma once


class INode {
public:
    virtual Tensor evaluate() const = 0;
};


class InputData : public INode {
private:
    Tensor tensor_;

public:
    InputData(const Tensor& tensor) : tensor_(tensor) {}

    virtual ~InputData() = default;
    virtual Tensor evaluate() const override {
        return tensor_;
    }
    
    void setTensor(const Tensor& tensor) {
        tensor_ = tensor;
    }
};


class BinaryOperation : public INode {
protected:
    std::shared_ptr<INode> lhs_, rhs_;
    Tensor lhs_tensor_, rhs_tensor_;
    std::vector<INode*> args_;
    bool lhs_is_node_, rhs_is_node_; // 1 -- is node,
                                     // 0 -- is tensor
public:
    BinaryOperation(const std::shared_ptr<INode> lhs, const std::shared_ptr<INode> rhs)
        : lhs_(lhs), rhs_(rhs), lhs_is_node_(1), rhs_is_node_(1) {
        args_.push_back(lhs.get());
        args_.push_back(rhs.get());
    }
    BinaryOperation(const std::shared_ptr<INode> lhs, const Tensor& rhs_tensor)
        : lhs_(lhs), rhs_tensor_(rhs_tensor), lhs_is_node_(1), rhs_is_node_(0) {
        args_.push_back(lhs.get());
    }
    BinaryOperation(const Tensor& lhs_tensor, const std::shared_ptr<INode> rhs)
        : rhs_(rhs), lhs_tensor_(lhs_tensor), lhs_is_node_(0), rhs_is_node_(1) {
        args_.push_back(rhs.get());
    }
    BinaryOperation(const Tensor& lhs_tensor, const Tensor& rhs_tensor)
        : lhs_tensor_(lhs_tensor), rhs_tensor_(rhs_tensor), lhs_is_node_(0), rhs_is_node_(0) {}

    virtual ~BinaryOperation() = default;
    virtual Tensor evaluate() const override = 0;
};


class UnaryOperation : public INode {
protected:
    std::shared_ptr<INode> arg_;
    std::vector<INode*> args_;
    Tensor tensor_;
    bool is_node_; // 1 -- is node,
                   // 0 -- is tensor
public:
    UnaryOperation(const std::shared_ptr<INode> arg): arg_(arg), is_node_(1) {
        args_.push_back(arg.get());
    }
    UnaryOperation(const Tensor& tensor): tensor_(tensor), is_node_(0) {}

    virtual ~UnaryOperation() = default;
    virtual Tensor evaluate() const override = 0;
};