#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
#include <THC/THC.h>
#include <iostream>

// C++ interface
// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> conv_forward(torch::Tensor input,
                                   torch::Tensor weights,
                                   torch::Tensor bias,
                                   int64_t kW, int64_t kH,
                                   int64_t dW, int64_t dH,
                                   int64_t padW, int64_t padH, bool is_bias) {

    CHECK_INPUT(input);
    CHECK_INPUT(weights);
    CHECK_INPUT(bias);

    // std::cout<<output.dim()<<std::endl;
    int64_t batch_size = input.size(0);
    int64_t nInputPlane = input.size(1);
    int64_t inputHeight = input.size(2);
    int64_t inputWidth = input.size(3);

    int64_t nOutputPlane = weights.size(0);
    int64_t outputHeight = (inputHeight + 2*padH - kH) / dH + 1;
    int64_t outputWidth = (inputWidth + 2*padW - kW) / dW + 1;

    torch::Tensor output = torch::zeros(torch::IntArrayRef({batch_size, nOutputPlane, outputHeight, outputWidth})).cuda();
    torch::Tensor columns = torch::zeros(torch::IntArrayRef({nInputPlane*kW*kH, outputHeight*outputWidth})).cuda();
    torch::Tensor ones = torch::ones(torch::IntArrayRef({1, outputHeight*outputWidth})).cuda();

    // after reshape, weights conv with columns
    weights = weights.reshape(torch::IntArrayRef({nOutputPlane, nInputPlane*kW*kH})).cuda();
    bias = bias.reshape(torch::IntArrayRef({nOutputPlane, 1})).cuda();

    for(int elt = 0; elt < batch_size; elt++){
        torch::Tensor input_n = input[elt];

        if(is_bias){
            output[elt].add_(bias.mm(ones).reshape(torch::IntArrayRef({nOutputPlane, outputHeight, outputWidth})).cuda(), 1);
        }

        // columns.dim: (inplanes * kW * kH) * (outHeight * outWidth)
        columns = torch::im2col(input_n.clone(), /*kernel_size=*/torch::IntArrayRef({kW, kH}), 
                                                 /*dilation=*/torch::IntArrayRef({1, 1}),
                                                 /*padding=*/torch::IntArrayRef({padW, padH}), 
                                                 /*stride=*/torch::IntArrayRef({dW, dH})).cuda();

        // weights.dim: outplanes * inplanes * kW * kH, conv(weights, coloumns)
        output[elt].add_(weights.mm(columns).reshape(torch::IntArrayRef({nOutputPlane, outputHeight, outputWidth})).cuda(), 1);
    }
    return {output};
}

torch::Tensor backward_gradInput(torch::Tensor input,
                                    torch::Tensor gradOutput,
                                    torch::Tensor weights,
                                    int64_t kW, int64_t kH,
                                    int64_t dW, int64_t dH,
                                    int64_t padW, int64_t padH){

    int64_t batch_size = input.size(0);
    int64_t nInputPlane = input.size(1);
    int64_t inputHeight = input.size(2);
    int64_t inputWidth = input.size(3);

    int64_t nOutputPlane = gradOutput.size(1);
    int64_t outputHeight = gradOutput.size(2);
    int64_t outputWidth = gradOutput.size(3);

    torch::Tensor gradInput = torch::zeros(torch::IntArrayRef({batch_size, nInputPlane, inputHeight, inputWidth})).cuda();
    torch::Tensor gradColumns = torch::zeros(torch::IntArrayRef({nInputPlane * kH * kW, outputHeight * outputWidth})).cuda();

    /* weight reshape to (inputPlanes * kW * kH) * outputplanes */
    torch::Tensor weights_ = weights.clone();
    weights = weights.reshape(torch::IntArrayRef({nOutputPlane, nInputPlane*kW*kH})).t().cuda();

    for(int elt = 0; elt < batch_size; elt++){
        torch::Tensor gradInput_n = gradInput[elt];
        torch::Tensor gradOutput_n = gradOutput[elt];

        // gradOutput_n.dim: nOutputPlane * (outputHeight*outputWidth)
        gradOutput_n = gradOutput_n.reshape(torch::IntArrayRef({nOutputPlane, outputHeight*outputWidth})).cuda();

        gradColumns = weights.mm(gradOutput_n).cuda();

        gradInput[elt].add_(torch::col2im(gradColumns.clone(), /*output_size=*/torch::IntArrayRef({inputHeight, inputWidth}),
                                                               /*kernel_size=*/torch::IntArrayRef({kW, kH}),
                                                               /*dilation=*/torch::IntArrayRef({1, 1}),
                                                               /*padding=*/torch::IntArrayRef({padW, padH}),
                                                               /*stride=*/torch::IntArrayRef({dW, dH})).cuda(), 1);
    }

    return gradInput;
}

std::vector<torch::Tensor> backward_gradParameters(torch::Tensor input,
                                                   torch::Tensor gradOutput,
                                                   torch::Tensor weights,
                                                   int64_t kW, int64_t kH,
                                                   int64_t dW, int64_t dH,
                                                   int64_t padW, int64_t padH,
                                                   bool is_bias){

    int64_t batch_size = input.size(0);
    int64_t nInputPlane = input.size(1);
    int64_t inputHeight = input.size(2);
    int64_t inputWidth = input.size(3);

    int64_t nOutputPlane = gradOutput.size(1);
    int64_t outputHeight = gradOutput.size(2);
    int64_t outputWidth = gradOutput.size(3);

    torch::Tensor gradWeights = torch::zeros(torch::IntArrayRef({weights.size(0), weights.size(1), 
                                                                 weights.size(2), weights.size(3)})).cuda();
    torch::Tensor gradBias = torch::zeros(torch::IntArrayRef({nOutputPlane})).cuda();
    torch::Tensor ones = torch::ones(torch::IntArrayRef({outputHeight*outputWidth, 1})).cuda();

    torch::Tensor columns = torch::zeros(torch::IntArrayRef({nInputPlane*kW*kH, outputHeight*outputWidth})).cuda();

    for(int elt = 0; elt < batch_size; elt++){
        torch::Tensor gradOutput_n = gradOutput[elt];
        gradOutput_n = gradOutput_n.reshape(torch::IntArrayRef({nOutputPlane, outputHeight*outputWidth})).cuda();

        // columns.dim: (inplanes * kW * kH) * (outHeight * outWidth)
        columns = torch::im2col(input[elt].clone(), /*kernel_size=*/torch::IntArrayRef({kW, kH}), 
                                                    /*dilation=*/torch::IntArrayRef({1, 1}),
                                                    /*padding=*/torch::IntArrayRef({padW, padH}), 
                                                    /*stride=*/torch::IntArrayRef({dW, dH})).t().cuda();
        gradWeights.add_(gradOutput_n.mm(columns).reshape(torch::IntArrayRef({nOutputPlane, nInputPlane, kW, kH})).cuda(), 1);

        if(is_bias){
            gradBias.add_(gradOutput_n.mm(ones).reshape(torch::IntArrayRef({nOutputPlane})), 1);
        }

    }
    return {gradWeights, gradBias};
}


std::vector<torch::Tensor> conv_backward(torch::Tensor input,
                                    torch::Tensor gradOutput,
                                    torch::Tensor weights,
                                    int64_t kW, int64_t kH,
                                    int64_t dW, int64_t dH,
                                    int64_t padW, int64_t padH,
                                    bool is_bias) {

    CHECK_INPUT(gradOutput);
    CHECK_INPUT(weights);
    CHECK_INPUT(input);

    torch::Tensor gradInput = backward_gradInput(input, gradOutput, weights, kW, kH, dW, dH, padW, padH);
    std::vector<torch::Tensor> gradParas = backward_gradParameters(input, gradOutput, weights, kW, kH, dW, dH, padW, padH, is_bias);

    torch::Tensor gradWeights = gradParas[0];
    torch::Tensor gradBias = gradParas[1];

    return {gradInput, gradWeights, gradBias};

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_forward, "conv forward (CUDA)");
  m.def("backward", &conv_backward, "conv backward (CUDA)");
}
