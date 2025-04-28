#include "Tensor.h"

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
    
    virtual Tensor evaluate() const override {
        return tensor_;
    }
    
    void setTensor(const Tensor& tensor) {
        tensor_ = tensor;
    }
};

/*
class IOperation : public INode {
public:
    virtual void setArgs(const std::vector<INode*>& args) = 0;
    virtual const std::vector<INode*>& getArgs() const = 0;
};
*/

class BinaryOperation : public INode {
protected:
    std::shared_ptr<INode> lhs_;
    Tensor rhs_;
    std::vector<INode*> args_;

public:
    BinaryOperation(const std::shared_ptr<INode> lhs, const Tensor& rhs): lhs_(lhs), rhs_(rhs) {
        args_.push_back(lhs.get());
    }

    virtual ~BinaryOperation() = default;
    virtual Tensor evaluate() const override = 0;
};


class UnaryOperation : public INode {
protected:
    std::shared_ptr<INode> arg_;
    std::vector<INode*> args_;

public:
    UnaryOperation(const std::shared_ptr<INode> arg): arg_(arg) {
        args_.push_back(arg.get());
    }

    virtual ~UnaryOperation() = default;
    virtual Tensor evaluate() const override = 0;
};
