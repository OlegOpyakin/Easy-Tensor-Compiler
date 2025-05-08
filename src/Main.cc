#include "NeuralNetwork.h"
#include <algorithm>
#include <iterator>
#include <simd/vector.h>

#define EPSILON 1e-6

bool ApproxEqual(float a, float b){
    if ((a <= b + EPSILON) and (a >= b - EPSILON)) return true;
    return false;
}

int main() try{
    std::cout << "*** E.T.C. (Easy Tensor Compiler) ***" << std::endl;
    NeuralNetwork nn;

    std::vector<float> input_values = {
        // Channel 0
        0.1f, 0.2f,
        0.3f, 0.4f,
        
        // Channel 1
        0.5f, 0.6f,
        0.7f, 0.8f,
        
        // Channel 2
        0.9f, 1.0f,
        1.1f, 1.2f
    };

    std::vector<float> expected_values = {
        // Channel 0
        0.2f, 0.3f,
        0.4f, 0.5f,
        
        // Channel 1
        0.6f, 0.7f,
        0.8f, 0.9f,
        
        // Channel 2
        1.0f, 1.1f,
        1.2f, 1.3f
    };

    std::vector<size_t> input_shape = {1, 3, 2, 2};
    Tensor input(input_shape, input_values);
    std::cout << "Input tensor: " << input << "\n";
    
    const auto& input_node = std::make_shared<InputData>(input);
    
    std::vector<float> bias_values(12, 0.1f); 
    std::vector<size_t> bias_shape = {1, 3, 2, 2};
    Tensor bias(bias_shape, bias_values);
    std::cout << "Bias tensor: " << bias << "\n";
    const auto& bias_node = std::make_shared<InputData>(bias);
    
    std::cout << "\nBuilding neural network with Addition -> Substraction\n";
    
    const auto& add_op = std::make_shared<ScalarAddOperation>(input_node, bias_node);
    nn.addOp(add_op);
    
    std::cout << "Running inference...\n";
    Tensor output = nn.infer();
    std::cout << "Output tensor: " << output << "\n";

    for(int i = 0; i < output.GetData().size(); ++i){
        assert(fabs(output.GetData()[i] - expected_values[i]) < EPSILON && "Tensor add error");
    }

    std::cout << "\nTest completed successfully!" << std::endl;
    return 0;
}
catch (const std::exception& e){
    std::cerr << "Error: " << e.what() << "\n";
}
catch (...){
    std::cerr << "Unknown exception\n";
}