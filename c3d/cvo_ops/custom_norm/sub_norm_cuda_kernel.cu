#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

template <typename scalar_t>
__global__ void sub_norm_cuda_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> x1,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> x2,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> y) {

  const int n2 = x2.size(2);
  const int n1 = x1.size(2);
  const int c = x1.size(1);

  // // cannot parallize across channels, because it will case modifying the the location by multiple threads at the same time
  // const int inc12 = blockIdx.x * blockDim.x + threadIdx.x;
  
  // const int ic = inc12 / (n1*n2);
  // const int in12 = inc12 % (n1*n2);
  // const int in1 = in12 / n2;
  // const int in2 = in12 % n2;

  // if (inc12 < n1 * n2 * c ){
  //   y[blockIdx.y][in1][in2] += (x1[blockIdx.y][ic][in1] - x2[blockIdx.y][ic][in2]) * (x1[blockIdx.y][ic][in1] - x2[blockIdx.y][ic][in2]) ;
  // }

  const int in12 = blockIdx.x * blockDim.x + threadIdx.x;
  
  const int in1 = in12 / n2;
  const int in2 = in12 % n2;
  

  if (in12 < n1 * n2 ){
    for (int ic = 0; ic < c; ic++){
      y[blockIdx.y][in1][in2] += (x1[blockIdx.y][ic][in1] - x2[blockIdx.y][ic][in2]) * (x1[blockIdx.y][ic][in1] - x2[blockIdx.y][ic][in2]) ;
    }
  }

}

// template <typename scalar_t>
// __global__ void sub_norm_cuda_backward_kernel(
//   torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> dx1,
//   torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> dx2,
//   const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> dy, 
//   const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> x1,
//   const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> x2) {

//   const int n2 = dx2.size(2);
//   const int n1 = dx1.size(2);
//   const int c = dx1.size(1);

//   const int inc12 = blockIdx.x * blockDim.x + threadIdx.x;
  
//   const int ic = inc12 / (n1*n2);
//   const int in12 = inc12 % (n1*n2);
//   const int in1 = in12 / n2;
//   const int in2 = in12 % n2;
  
//   if (inc12 < n1 * n2 * c ){
//     dx1[blockIdx.y][ic][in1] += dy[blockIdx.y][in1][in2] * 2 * (x1[blockIdx.y][ic][in1] - x2[blockIdx.y][ic][in2]);
//     dx2[blockIdx.y][ic][in2] -= dy[blockIdx.y][in1][in2] * 2 * (x1[blockIdx.y][ic][in1] - x2[blockIdx.y][ic][in2]);
//   }

// }
// } // namespace

template <typename scalar_t>
__global__ void sub_norm_cuda_backward_kernel_dx(
  torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> dx1,
  torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> dx2,
  const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> dy, 
  const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> x1,
  const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> x2) {

  const int n2 = x2.size(2);
  const int n1 = x1.size(2);
  const int c = x1.size(1);

  const int inc12 = blockIdx.x * blockDim.x + threadIdx.x;
  const int in12 = inc12 / c;
  const int ic = inc12 % c;

  if (in12 < n1){
    const int in1 = in12;
    if (in1 < n1 ){
      for (int in2 = 0; in2 < n2; in2++){
        dx1[blockIdx.y][ic][in1] += dy[blockIdx.y][in1][in2] * 2 * (x1[blockIdx.y][ic][in1] - x2[blockIdx.y][ic][in2]);
      }
    }
  }
  else{
    const int in2 = in12 - n1;
    if (in2 < n2){
      for (int in1 = 0; in1 < n1; in1++){
        dx2[blockIdx.y][ic][in2] -= dy[blockIdx.y][in1][in2] * 2 * (x1[blockIdx.y][ic][in1] - x2[blockIdx.y][ic][in2]);
      }
    }
  }
}

template <typename scalar_t>
__global__ void sub_norm_cuda_backward_kernel_dx1(
  torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> dx1,
  const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> dy, 
  const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> x1,
  const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> x2) {

  const int n2 = x2.size(2);
  const int n1 = x1.size(2);
  const int c = x1.size(1);

  const int inc1 = blockIdx.x * blockDim.x + threadIdx.x;
  
  const int ic = inc1 / n1;
  const int in1 = inc1 % n1;
  
  if (inc1 < n1 * c ){
    for (int in2 = 0; in2 < n2; in2++){
      dx1[blockIdx.y][ic][in1] += dy[blockIdx.y][in1][in2] * 2 * (x1[blockIdx.y][ic][in1] - x2[blockIdx.y][ic][in2]);
    }
  }
}

template <typename scalar_t>
__global__ void sub_norm_cuda_backward_kernel_dx2(
  torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> dx2,
  const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> dy, 
  const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> x1,
  const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> x2) {

  const int n2 = x2.size(2);
  const int n1 = x1.size(2);
  const int c = x1.size(1);

  const int inc2 = blockIdx.x * blockDim.x + threadIdx.x;
  
  const int ic = inc2 / n2;
  const int in2 = inc2 % n2;
  
  if (inc2 < n2 * c ){
    for (int in1 = 0; in1 < n1; in1++){
      dx2[blockIdx.y][ic][in2] -= dy[blockIdx.y][in1][in2] * 2 * (x1[blockIdx.y][ic][in1] - x2[blockIdx.y][ic][in2]);
    }
  }
}
} // namespace

torch::Tensor sub_norm_cuda_forward(
    torch::Tensor x1,
    torch::Tensor x2) {

  const auto batch_size = x1.size(0);
  const auto channel_size = x1.size(1);
  const auto n1 = x1.size(2);
  const auto n2 = x2.size(2);

  auto options = torch::TensorOptions().dtype(x1.dtype()).layout(torch::kStrided).device(x1.device()).requires_grad(true);
  auto y = torch::zeros({batch_size, n1, n2}, options);

  // printf("x1 device: %d \n", x1.device().type()); 
  // printf("x1 index: %d \n", x1.device().index()); 

  const int threads = 1024;
  // cannot parallize across channels, because it will case modifying the the location by multiple threads at the same time
  // const dim3 blocks((n1 * n2 * channel_size + threads - 1) / threads, batch_size);
  const dim3 blocks((n1 * n2  + threads - 1) / threads, batch_size);
  // const dim3 blocks(1, 1);

  int device_id = x1.device().index();
  cudaSetDevice(device_id);

  // AT_DISPATCH_FLOATING_TYPES AT_DISPATCH_ALL_TYPES_AND_HALF
 AT_DISPATCH_FLOATING_TYPES(x1.type(), "sub_norm_forward_cuda", ([&] {
    sub_norm_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
      x1.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
      x2.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
      y.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>());
  }));
  cudaDeviceSynchronize();  
  return y;
}

std::vector<torch::Tensor> sub_norm_cuda_backward(
    torch::Tensor dy, 
    torch::Tensor x1,
    torch::Tensor x2) {
  const auto batch_size = dy.size(0);
  const auto channel_size = x1.size(1);
  const auto n1 = dy.size(1);
  const auto n2 = dy.size(2);

  auto dx1 = torch::zeros({batch_size, channel_size, n1}, x1.device());
  auto dx2 = torch::zeros({batch_size, channel_size, n2}, x1.device());

  const int threads = 1024;
  // The backward function can parallize channels because it does not cause same-time modification of the same location
  // But n1 and n2 will conflict with each other when parallelized together 
  // because dx1 (and dx2) may be modified at the same location by multiple thread with the same in1 or in2 index
  // const dim3 blocks((n1 * n2 * channel_size + threads - 1) / threads, batch_size);

  // AT_DISPATCH_FLOATING_TYPES(dy.type(), "sub_norm_forward_cuda", ([&] {
  //   sub_norm_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
  //     dx1.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
  //     dx2.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
  //     dy.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(), 
  //     x1.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
  //     x2.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>());
  // }));

  int device_id = x1.device().index();
  cudaSetDevice(device_id);

  const dim3 blocks_dx12(( (n1+n2) * channel_size + threads - 1) / threads, batch_size);
  // AT_DISPATCH_ALL_TYPES_AND_HALF
  AT_DISPATCH_FLOATING_TYPES(dy.type(), "sub_norm_backward_cuda_dx", ([&] {
    sub_norm_cuda_backward_kernel_dx<scalar_t><<<blocks_dx12, threads>>>(
      dx1.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
      dx2.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
      dy.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(), 
      x1.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
      x2.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>());
  }));
  cudaDeviceSynchronize();  


  // const dim3 blocks_dx1((n1 * channel_size + threads - 1) / threads, batch_size);
  // // AT_DISPATCH_FLOATING_TYPES
  // AT_DISPATCH_ALL_TYPES_AND_HALF(dy.type(), "sub_norm_backward_cuda_dx1", ([&] {
  //   sub_norm_cuda_backward_kernel_dx1<scalar_t><<<blocks_dx1, threads>>>(
  //     dx1.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
  //     dy.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(), 
  //     x1.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
  //     x2.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>());
  // }));
  // cudaDeviceSynchronize();  

  // const dim3 blocks_dx2((n2 * channel_size + threads - 1) / threads, batch_size);

  // // AT_DISPATCH_FLOATING_TYPES
  // AT_DISPATCH_ALL_TYPES_AND_HALF(dy.type(), "sub_norm_backward_cuda_dx2", ([&] {
  //   sub_norm_cuda_backward_kernel_dx2<scalar_t><<<blocks_dx2, threads>>>(
  //     dx2.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
  //     dy.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(), 
  //     x1.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
  //     x2.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>());
  // }));
  // cudaDeviceSynchronize();  

  return {dx1, dx2};
}
