#include "ScalarAddOperation.h"
#include "ScalarSubOperation.h"
#include "ScalarMulOperation.h"
#include "MatMulOperation.h"
#include "ReLUOperation.h"
#include "SoftmaxOperation.h"
#include <unordered_set>

#pragma once

class NeuralNetwork {
private:
    std::vector<std::shared_ptr<INode>> operations_;

public:
    // Add operation
    std::shared_ptr<INode> addOp(std::shared_ptr<INode> op) {
        operations_.push_back(op);
        return op;
    }
    
    Tensor infer() {
        if (operations_.empty()) return Tensor();
        return operations_.back()->evaluate();
    }
    
    // Get all operations in the network
    const std::vector<std::shared_ptr<INode>>& getOperations() const { return operations_; }
    
    // Clear all operations
    void clear() { operations_.clear(); }
    
};