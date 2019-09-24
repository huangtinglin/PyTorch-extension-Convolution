from torch.utils.cpp_extension import load
conv_cuda = load('conv_cuda', sources=['conv_cuda.cpp'], verbose=True)
help(conv_cuda)

