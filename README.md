# Implemented convolution based on CUDA extensions in PyTorch

A convolution implementation based on cuda extension for PyTorch. The source code reference to the PyTorch's inefficient implementation [here](https://github.com/pytorch/pytorch/blob/master/aten/src/THCUNN/generic/SpatialConvolutionMM.cu). See [here](http://pytorch.org/tutorials/advanced/cpp_extension.html) for the accompanying tutorial.

- Build CUDA extensions by going into the `conv_cuda/` folder and executing `python setup.py install`,
- JIT-compile CUDA extensions by going into the `conv_cuda/` folder and calling `python jit.py`, which will JIT-compile the extension and load it,
- Check the result of the convolution by running `python test.py`



