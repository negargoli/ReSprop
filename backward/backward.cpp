#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <ATen/cuda/CUDAConfig.h>
#include <ATen/ArrayRef.h>
#include <iostream>
#include <cuda_runtime.h>
#include <time.h>
#include <sys/time.h>

at::Tensor conv_input(std::vector<int64_t>input_size_p, const at::Tensor& grad_output, const at::Tensor& weight, std::vector<int64_t> stride_p,std::vector<int64_t> padding_p, std::vector<int64_t> dilation_p, int64_t groups)
             
{

    cudaEvent_t start1, stop;
    at::ArrayRef<int64_t> stride{stride_p};
    at::ArrayRef<int64_t> padding{padding_p};
    at::ArrayRef<int64_t> dilation{dilation_p};
    at::ArrayRef<int64_t> input_size{input_size_p};
    bool deterministic=true;
    bool benchmark=false;
    float time = 0;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop);
    cudaEventRecord(start1, 0);
    at::Tensor result = (cudnn_convolution_backward_input(input_size, grad_output, weight,padding, stride, dilation, groups, benchmark, deterministic));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start1, stop);
    //printf("conv_input:  %f ms \n", time);
    return result;

}

at::Tensor conv_weight(std::vector<int64_t> weight_size_p, const at::Tensor& grad_output, const at::Tensor& input, std::vector<int64_t> stride_p, std::vector<int64_t> padding_p, std::vector<int64_t> dilation_p, int64_t groups)

{
    cudaEvent_t start1, stop;
    at::ArrayRef<int64_t> stride{stride_p};
    at::ArrayRef<int64_t> padding{padding_p};
    at::ArrayRef<int64_t> dilation{dilation_p};
    at::ArrayRef<int64_t> weight_size{weight_size_p};
    bool deterministic=true;
    bool benchmark=false;
    float time = 0;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop);
    cudaEventRecord(start1, 0);
    at::Tensor result = (cudnn_convolution_backward_weight(weight_size, input, grad_output,padding, stride, dilation, groups, benchmark, deterministic));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start1, stop);
    //printf("conv_weight:  %f ms \n", time);
    return result;
	
}
at::Tensor conv2d_forward( at::Tensor& input, at::Tensor& weight,at::Tensor& bias,std::vector<int64_t> stride_p, std::vector<int64_t> padding_p,std::vector<int64_t> dilation_p, int64_t groups)
{
   cudaEvent_t start1, stop;
   at::ArrayRef<int64_t> stride{stride_p};
   at::ArrayRef<int64_t> padding{padding_p};
   at::ArrayRef<int64_t> dilation{dilation_p};
   float time = 0;
   cudaEventCreate(&start1);
   cudaEventCreate(&stop);
   cudaEventRecord(start1, 0);
   at::Tensor result = (at::conv2d(input,weight,bias,stride,padding,dilation,groups));
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&time, start1, stop);
   //printf("conv_forward:  %f ms \n", time);
   return result;

}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("weight", &conv_weight, "conv weight");
  m.def("input", &conv_input, "conv input");
  m.def("forwardconv", &conv2d_forward, "conv forward");
}
