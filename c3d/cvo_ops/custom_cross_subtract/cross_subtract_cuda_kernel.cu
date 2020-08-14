#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

template <typename scalar_t>
__global__ void cross_subtract_cuda_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> x1,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> x2,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> y) {

  const int n2 = x2.size(2);
  const int n1 = x1.size(2);
  const int c = x1.size(1);

  const int inc12 = blockIdx.x * blockDim.x + threadIdx.x;
  
  const int in12 = inc12 / c;
  const int inc = inc12 % c;
  const int in1 = in12 / n2;
  const int in2 = in12 % n2;

  if (inc12 < n1 * n2 * c ){
    y[blockIdx.y][in1][in2][inc] = x2[blockIdx.y][inc][in2] - x1[blockIdx.y][inc][in1];
  }

}

}


torch::Tensor cross_subtract_cuda_forward(
    torch::Tensor x1,
    torch::Tensor x2) {

  const auto batch_size = x1.size(0);
  const auto channel_size = x1.size(1);
  const auto n1 = x1.size(2);
  const auto n2 = x2.size(2);

  auto options = torch::TensorOptions().dtype(x1.dtype()).layout(torch::kStrided).device(x1.device()).requires_grad(true);
  auto y = torch::zeros({batch_size, n1, n2, channel_size}, options);

  // printf("x1 device: %d \n", x1.device().type()); 
  // printf("x1 index: %d \n", x1.device().index()); 

  const int threads = 1024;
  const dim3 blocks((n1 * n2 * channel_size  + threads - 1) / threads, batch_size);

  int device_id = x1.device().index();
  cudaSetDevice(device_id);

  AT_DISPATCH_FLOATING_TYPES(x1.type(), "cross_subtract_forward_cuda", ([&] {
    cross_subtract_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
      x1.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
      x2.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
      y.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>());
  }));
  cudaDeviceSynchronize();  
  return y;
}
