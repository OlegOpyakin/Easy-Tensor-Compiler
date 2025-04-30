#include <gtest/gtest.h>
#include "NeuralNetwork.h"
#include <random>
#include <ctime>
#include <numeric>

#define EPSILON 1e-2

class TestBinaryOperation : public ::testing::Test{
protected:
    Tensor* t1;
    Tensor* t2;
    std::vector<float> expected_values;

    void SetUp() override {
        std::vector<size_t> input_shape = {1, 3, 2, 2};
        std::vector<float> input_values = {
            0.1f, 0.2f,
            0.3f, 0.4f,
            0.5f, 0.6f,
            0.7f, 0.8f,
            0.9f, 1.0f,
            1.1f, 1.2f
        };

        t1 = new Tensor(input_shape, input_values);
    }

    void TearDown() override {
        delete t1;
        delete t2;
    }
};

class TestUnaryOperation : public ::testing::Test{
protected:
    Tensor* t1;
    std::vector<float> expected_values;

    void SetUp() override {
        std::vector<size_t> input_shape = {1, 3, 2, 2};
        std::vector<float> input_values = {
            0.1f, -0.2f,
            0.3f, -0.4f,
            -0.5f, 0.6f,
            0.7f, -0.8f,
            -0.9f, 1.0f,
            1.1f, 1.2f
        };

        t1 = new Tensor(input_shape, input_values);
    }

    void TearDown() override {
        delete t1;
    }
};


class TestNeuralNetwork : public ::testing::Test{
protected:
    Tensor* t1;
    Tensor* t2;
    std::vector<float> expected_values;

    void SetUp() override {
        std::vector<size_t> input_shape = {1, 3, 2, 2};
        std::vector<float> input_values = {
            0.1f, 0.2f,
            0.3f, 0.4f,
            0.5f, 0.6f,
            0.7f, 0.8f,
            0.9f, 1.0f,
            1.1f, 1.2f
        };

        t1 = new Tensor(input_shape, input_values);
    }

    void TearDown() override {
        delete t1;
        delete t2;
    }
};


// ------------------------------- TESTS BINARY OPS -------------------------------


TEST_F(TestBinaryOperation, ScalarAddOperation) {
    NeuralNetwork nn;

    std::vector<float> bias_values(12, 0.1f);
    std::vector<size_t> bias_shape = {1, 3, 2, 2};
        
    t2 = new Tensor(bias_shape, bias_values);
    
    expected_values = {
        0.2f, 0.3f,
        0.4f, 0.5f,
        0.6f, 0.7f,
        0.8f, 0.9f,
        1.0f, 1.1f,
        1.2f, 1.3f
    };

    const auto& input_node = std::make_shared<InputData>(*t1);
    const auto& add_node = std::make_shared<InputData>(*t2);

    const auto& add_op = std::make_shared<ScalarAddOperation>(input_node, add_node);
    nn.addOp(add_op);

    Tensor output = nn.infer();

    for(int i = 0; i < output.GetData().size(); ++i){
        EXPECT_TRUE(fabs(output.GetData()[i] - expected_values[i]) < EPSILON);
    }
}


TEST_F(TestBinaryOperation, ScalarSubOperation) {
    NeuralNetwork nn;

    std::vector<float> bias_values(12, -0.1f);
    std::vector<size_t> bias_shape = {1, 3, 2, 2};
        
    t2 = new Tensor(bias_shape, bias_values);

    expected_values = {
        0.2f, 0.3f,
        0.4f, 0.5f,
        0.6f, 0.7f,
        0.8f, 0.9f,
        1.0f, 1.1f,
        1.2f, 1.3f
    };

    const auto& input_node = std::make_shared<InputData>(*t1);
    const auto& sub_node = std::make_shared<InputData>(*t2);

    const auto& sub_op = std::make_shared<ScalarSubOperation>(input_node, sub_node);
    nn.addOp(sub_op);

    Tensor output = nn.infer();

    for(int i = 0; i < output.GetData().size(); ++i){
        EXPECT_TRUE(fabs(output.GetData()[i] - expected_values[i]) < EPSILON);
    }
}


TEST_F(TestBinaryOperation, ScalarMulOperation) {
    NeuralNetwork nn;
    
    std::vector<float> bias_values(12, 10.0f);
    std::vector<size_t> bias_shape = {1, 3, 2, 2};

    t2 = new Tensor(bias_shape, bias_values);

    expected_values = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
    };

    const auto& input_node = std::make_shared<InputData>(*t1);
    const auto& mul_node = std::make_shared<InputData>(*t2);

    const auto& mul_op = std::make_shared<ScalarMulOperation>(input_node, mul_node);
    nn.addOp(mul_op);

    Tensor output = nn.infer();

    for(int i = 0; i < output.GetData().size(); ++i){
        EXPECT_TRUE(fabs(output.GetData()[i] - expected_values[i]) < EPSILON);
    }
}


TEST_F(TestBinaryOperation, MatMulOperation) {
    NeuralNetwork nn;

    std::vector<float> bias_values = {
        10.0f, 1.0f,
        1.0f, 10.0f,
        10.0f, 1.0f,
        1.0f, 10.0f,
        10.0f, 1.0f,
        1.0f, 10.0f,
    };
    std::vector<size_t> bias_shape = {1, 3, 2, 2};
        
    t2 = new Tensor(bias_shape, bias_values);

    expected_values = {
        1.2f, 2.1f,
        3.4f, 4.3f,
        5.6f, 6.5f,
        7.8f, 8.7f,
        10.0f, 10.9f,
        12.2f, 13.1f
    };

    const auto& input_node = std::make_shared<InputData>(*t1);
    const auto& mul_node = std::make_shared<InputData>(*t2);

    const auto& mul_op = std::make_shared<MatMulOperation>(input_node, mul_node);
    nn.addOp(mul_op);

    Tensor output = nn.infer();

    for(int i = 0; i < output.GetData().size(); ++i){
        EXPECT_TRUE(fabs(output.GetData()[i] - expected_values[i]) < EPSILON);
    }
}


// ------------------------------- TESTS UNARY OPS -------------------------------


TEST_F(TestUnaryOperation, ReLUOperation) {
    NeuralNetwork nn;

    expected_values = {
        0.1f, 0.0f,
        0.3f, 0.0f,
        0.0f, 0.6f,
        0.7f, 0.0f,
        0.0f, 1.0f,
        1.1f, 1.2f
    };

    const auto& input_node = std::make_shared<InputData>(*t1);

    const auto& relu_op = std::make_shared<ReLUOperation>(input_node);
    nn.addOp(relu_op);

    Tensor output = nn.infer();

    for(int i = 0; i < output.GetData().size(); ++i){
        EXPECT_TRUE(fabs(output.GetData()[i] - expected_values[i]) < EPSILON);
    }
}


TEST_F(TestUnaryOperation, SoftmaxOperation) {
    NeuralNetwork nn;

    expected_values = {
        0.248, 0.224, 
        0.302, 0.224, 
        0.171, 0.312, 
        0.345, 0.171, 
        0.099, 0.271, 
        0.299, 0.331
    };

    Tensor t_result = *t1;

    const auto& input_node = std::make_shared<InputData>(*t1);

    const auto& relu_op = std::make_shared<ReLUOperation>(input_node);
    nn.addOp(relu_op);

    const auto& softmax_op = std::make_shared<SoftmaxOperation>(relu_op);
    nn.addOp(softmax_op);

    Tensor output = nn.infer();

    for(int i = 0; i < output.GetData().size(); ++i){
        EXPECT_TRUE(fabs(output.GetData()[i] - expected_values[i]) < EPSILON);
    }

    std::vector<float> to_sum;
    float layer_sum = 0.0f;

    for(int i = 0; i < output.GetData().size(); ++i){
        to_sum.push_back(output.GetData()[i]);

        if(i + 1 % 4 == 0){
            std::for_each(to_sum.rbegin(), to_sum.rend(), [&](int n) { layer_sum += n; });
            EXPECT_TRUE(fabs(layer_sum - 1) < 0.01);
            layer_sum = 0.0f;
            to_sum.clear();
        }
    }
}


// ------------------------------- TESTS NN -------------------------------


TEST_F(TestNeuralNetwork, NeuralNetwork) {
    NeuralNetwork nn;

    std::vector<float> bias_values(12, 0.1f);
    std::vector<size_t> bias_shape = {1, 3, 2, 2};
        
    t2 = new Tensor(bias_shape, bias_values);

    expected_values = {
        0.3f, 0.4f,
        0.5f, 0.6f,
        0.7f, 0.8f,
        0.9f, 1.0f,
        1.1f, 1.2f,
        1.3f, 1.4f
    };

    const auto& input_node = std::make_shared<InputData>(*t1);
    
    const auto& add1_op = std::make_shared<ScalarAddOperation>(input_node, *t2);
    nn.addOp(add1_op);

    const auto& add2_op = std::make_shared<ScalarAddOperation>(*t2, add1_op);
    nn.addOp(add2_op);

    Tensor output = nn.infer();

    for(int i = 0; i < output.GetData().size(); ++i){
        EXPECT_TRUE(fabs(output.GetData()[i] - expected_values[i]) < EPSILON);
    }
}


// ------------------------------- MAIN -------------------------------


int main(int argc, char **argv) try {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
catch (const std::exception& e){
    std::cerr << "Error: " << e.what() << "\n";
}
catch (...){
    std::cerr << "Unknown exception\n";
}
