from torch import nn
import torch
import torch.nn.functional as F
import conv_cuda


class ConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias, params):

        dW, dH, padW, padH, is_bias = int(params[0]), int(params[1]), int(params[2]), int(params[3]), int(params[4])
        kW, kH = weights.shape[2], weights.shape[3]

        outputs = conv_cuda.forward(input, weights, bias, kW, kH, dW, dH, padW, padH, is_bias)[0]

        variables = [input, weights, bias, params]
        ctx.save_for_backward(*variables)

        return outputs

    @staticmethod
    def backward(ctx, gradOutput):
        _ = torch.autograd.Variable(torch.zeros(5))

        input, weights, bias, params = ctx.saved_tensors

        dW, dH, padW, padH, is_bias = int(params[0]), int(params[1]), int(params[2]), int(params[3]), int(params[4])
        kW, kH = weights.shape[2], weights.shape[3]

        gradInput, gradWeight, gradBias = conv_cuda.backward(input, gradOutput, weights,
                                                             kW, kH, dW, dH, padW, padH, is_bias)
        return gradInput, gradWeight, gradBias, _


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, is_bias=True):
        super(Conv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.is_bias = is_bias

        self.params = torch.autograd.Variable(torch.Tensor([stride, stride, padding, padding, is_bias])).cuda()

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size).cuda())
        self.bias = nn.Parameter(torch.empty(out_channels).cuda())

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.is_bias:
            nn.init.constant_(self.bias, 0)

    def forward(self, input):
        return ConvFunction.apply(input, self.weight, self.bias, self.params)
