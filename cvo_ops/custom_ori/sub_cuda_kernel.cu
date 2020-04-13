#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

template <typename scalar_t>
__global__ void sub_cuda_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> x1,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> x2,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> y) {

  const int n2 = x2.size(2);
  const int n1 = x1.size(2);
  const int c = x1.size(1);

  const int inc12 = blockIdx.x * blockDim.x + threadIdx.x;
  
  const int ic = inc12 / (n1*n2);
  const int in12 = inc12 % (n1*n2);
  const int in1 = in12 / n2;
  const int in2 = in12 % n2;

  if (inc12 < n1 * n2 * c ){
    y[blockIdx.y][ic][in1][in2] = x1[blockIdx.y][ic][in1] - x2[blockIdx.y][ic][in2];
  }
}

template <typename scalar_t>
__global__ void sub_cuda_backward_kernel(
  torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> dx1,
  torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> dx2,
  const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> dy) {

  const int n2 = dx2.size(2);
  const int n1 = dx1.size(2);
  const int c = dx1.size(1);

  const int inc12 = blockIdx.x * blockDim.x + threadIdx.x;
  
  const int ic = inc12 / (n1*n2);
  const int in12 = inc12 % (n1*n2);
  const int in1 = in12 / n2;
  const int in2 = in12 % n2;

  
  if (inc12 < n1 * n2 * c ){
    dx1[blockIdx.y][ic][in1] += dy[blockIdx.y][ic][in1][in2];
    dx2[blockIdx.y][ic][in2] -= dy[blockIdx.y][ic][in1][in2];
  }
}
} // namespace

torch::Tensor sub_cuda_forward(
    torch::Tensor x1,
    torch::Tensor x2) {

  const auto batch_size = x1.size(0);
  const auto channel_size = x1.size(1);
  const auto n1 = x1.size(2);
  const auto n2 = x2.size(2);

  auto y = torch::zeros({batch_size, channel_size, n1, n2});

  const int threads = 1024;
  const dim3 blocks((n1 * n2 * channel_size + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(x1.type(), "sub_forward_cuda", ([&] {
    sub_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
      x1.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
      x2.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
      y.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>());
  }));

  return y;
}

std::vector<torch::Tensor> sub_cuda_backward(
    torch::Tensor dy) {
  const auto batch_size = dy.size(0);
  const auto channel_size = dy.size(1);
  const auto n1 = dy.size(2);
  const auto n2 = dy.size(3);

  auto dx1 = torch::zeros({batch_size, channel_size, n1});
  auto dx2 = torch::zeros({batch_size, channel_size, n2});

  const int threads = 1024;
  const dim3 blocks((n1 * n2 * channel_size + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(dy.type(), "sub_forward_cuda", ([&] {
    sub_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
      dx1.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
      dx2.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
      dy.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>());
  }));

  return {dx1, dx2};
}
