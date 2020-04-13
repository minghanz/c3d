#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <math.h>

namespace {

template <typename scalar_t>
__global__ void cvo_dense_samp_cuda_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> pts,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> pts_info,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> grid_source,
    const torch::PackedTensorAccessor<bool,4,torch::RestrictPtrTraits,size_t> grid_valid,
    const int neighbor_range, 
    const float ell,
    const bool ignore_ib, 
    const bool sqr,
    const float ell_basedist, 
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> y) {

  const auto N = pts.size(2);
  const auto C = pts_info.size(1);
  const auto B = grid_source.size(0);
  const auto H = grid_source.size(2);
  const auto W = grid_source.size(3);
  const int NN_sqrt = 2 * neighbor_range + 1;

  //dim3 block[N, NN, 1]
  const auto in = blockIdx.x * blockDim.x + threadIdx.x;  
  const int innh = blockIdx.y / NN_sqrt - neighbor_range;
  const int innw = blockIdx.y % NN_sqrt - neighbor_range;

  if (in < N ){
    const int u = pts[0][0][in];
    const int v = pts[0][1][in];
    int ib;
    if (ignore_ib){
      ib = 0;
    }
    else{
      ib = pts[0][2][in];
    }
    if (u+innw >= 0 && u+innw < W && v+innh >= 0 && v+innh < H){
      if (grid_valid[ib][0][v+innh][u+innw] > 0){

        float ell_apply;
        if (ell_basedist!= 0){
          float flat_z = pts_info[0][2][in];
          ell_apply = max(flat_z, ell_basedist) / ell_basedist * ell;
        }
        else{
          ell_apply = ell;
        }

        float d_cur = 0;
        for (int ic = 0; ic < C; ic++){
          d_cur += (pts_info[0][ic][in] - grid_source[ib][ic][v+innh][u+innw]) * (pts_info[0][ic][in] - grid_source[ib][ic][v+innh][u+innw]);
        }
        if (sqr){
          y[0][blockIdx.y][in] = exp( - d_cur / (2*ell_apply*ell_apply) ) ;
        }
        else{
          y[0][blockIdx.y][in] = exp( - sqrt(d_cur+1e-8) / ell_apply ) ;
        }
        
      }
    }
  }

}


template <typename scalar_t>
__global__ void cvo_dense_samp_cuda_backward_kernel_sqr_only(
  torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> dx1,
  torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> dx2,
  const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> dy, 
  const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> y, 
  const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> pts,
  const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> pts_info,
  const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> grid_source,
  const torch::PackedTensorAccessor<bool,4,torch::RestrictPtrTraits,size_t> grid_valid,
  const int neighbor_range, 
  const float ell, 
  const bool ignore_ib, 
  const bool sqr,
  const float ell_basedist, 
  const int inn) {
  // dx1: 1*C*N
  // dx2: B*C*H*W
  // dy: 1*NN*N

  const auto N = pts.size(2);
  const auto C = pts_info.size(1);
  const auto B = grid_source.size(0);
  const auto H = grid_source.size(2);
  const auto W = grid_source.size(3);
  const auto NN = (2*neighbor_range+1)*(2*neighbor_range+1);
  const auto NN_sqrt = 2 * neighbor_range + 1;

  //dim3 block[N, C, 1] 
  if (inn < NN){
    const int in = blockIdx.x * blockDim.x + threadIdx.x;
    const int innh = inn / NN_sqrt - neighbor_range;
    const int innw = inn % NN_sqrt - neighbor_range;

    if (in < N ){
      const int u = pts[0][0][in];
      const int v = pts[0][1][in];
      int ib;
      if (ignore_ib){
        ib = 0;
      }
      else{
        ib = pts[0][2][in];
      }
      if (u+innw >= 0 && u+innw < W && v+innh >= 0 && v+innh < H){
        if (grid_valid[ib][0][v+innh][u+innw] > 0){

          float ell_apply;
          if (ell_basedist!= 0){
            float flat_z = pts_info[0][2][in];
            ell_apply = max(flat_z, ell_basedist) / ell_basedist * ell;
          }
          else{
            ell_apply = ell;
          }

          dx1[0][blockIdx.y][in] += dy[0][inn][in] * y[0][inn][in] * (grid_source[ib][blockIdx.y][v+innh][u+innw] - pts_info[0][blockIdx.y][in]) / (ell_apply*ell_apply);
          dx2[ib][blockIdx.y][v+innh][u+innw] -= dy[0][inn][in] * y[0][inn][in] * (grid_source[ib][blockIdx.y][v+innh][u+innw] - pts_info[0][blockIdx.y][in]) / (ell_apply*ell_apply);
        }
      }
    }
  }
  
}

template <typename scalar_t>
__global__ void cvo_dense_samp_cuda_backward_kernel(
  torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> dx1,
  torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> dx2,
  const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> dy, 
  const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> pts,
  const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> pts_info,
  const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> grid_source,
  const torch::PackedTensorAccessor<bool,4,torch::RestrictPtrTraits,size_t> grid_valid,
  const int neighbor_range, 
  const float ell, 
  const bool ignore_ib, 
  const bool sqr,
  const float ell_basedist, 
  const int inn) {
  // dx1: 1*C*N
  // dx2: B*C*H*W
  // dy: 1*NN*N

  const auto N = pts.size(2);
  const auto C = pts_info.size(1);
  const auto B = grid_source.size(0);
  const auto H = grid_source.size(2);
  const auto W = grid_source.size(3);
  const auto NN = (2*neighbor_range+1)*(2*neighbor_range+1);
  const auto NN_sqrt = 2 * neighbor_range + 1;

  //dim3 block[N, C, 1] 
  if (inn < NN){
    const int in = blockIdx.x * blockDim.x + threadIdx.x;
    const int innh = inn / NN_sqrt - neighbor_range;
    const int innw = inn % NN_sqrt - neighbor_range;

    if (in < N ){
      const int u = pts[0][0][in];
      const int v = pts[0][1][in];
      int ib;
      if (ignore_ib){
        ib = 0;
      }
      else{
        ib = pts[0][2][in];
      }
      if (u+innw >= 0 && u+innw < W && v+innh >= 0 && v+innh < H){
        if (grid_valid[ib][0][v+innh][u+innw] > 0){

          float ell_apply;
          if (ell_basedist!= 0){
            float flat_z = pts_info[0][2][in];
            ell_apply = max(flat_z, ell_basedist) / ell_basedist * ell;
          }
          else{
            ell_apply = ell;
          }

          float d_cur = 0;
          for (int ic = 0; ic < C; ic++){
            d_cur += (pts_info[0][ic][in] - grid_source[ib][ic][v+innh][u+innw]) * (pts_info[0][ic][in] - grid_source[ib][ic][v+innh][u+innw]);
          }

          float y_cur;
          if (sqr){
            y_cur = exp( - d_cur / (2*ell_apply*ell_apply) );
            for (int ic = 0; ic < C; ic++){
              dx1[0][ic][in] += dy[0][inn][in] * y_cur * (grid_source[ib][ic][v+innh][u+innw] - pts_info[0][ic][in]) / (ell_apply*ell_apply);
              dx2[ib][ic][v+innh][u+innw] -= dy[0][inn][in] * y_cur * (grid_source[ib][ic][v+innh][u+innw] - pts_info[0][ic][in]) / (ell_apply*ell_apply);
            }
          }
          else{
            d_cur = sqrt(d_cur + 1e-8);
            y_cur = exp( - d_cur / ell_apply );
            for (int ic = 0; ic < C; ic++){
              dx1[0][ic][in] += dy[0][inn][in] * y_cur * (grid_source[ib][ic][v+innh][u+innw] - pts_info[0][ic][in]) / ell_apply / d_cur;
              dx2[ib][ic][v+innh][u+innw] -= dy[0][inn][in] * y_cur * (grid_source[ib][ic][v+innh][u+innw] - pts_info[0][ic][in]) / ell_apply / d_cur;
            }

          }
        }
      }
    }
  }
  
}

} // namespace

torch::Tensor cvo_dense_samp_cuda_forward(
    torch::Tensor pts,
    torch::Tensor pts_info, 
    torch::Tensor grid_source, 
    torch::Tensor grid_valid, 
    int neighbor_range,
    float ell, 
    bool ignore_ib, 
    bool sqr,
    float ell_basedist
    ) {
    // pts: 1*2*N, pts_info: 1*C*N, grid_source: B*C*H*W (C could be xyz, rgb, ...), 
    // grid_valid: B*1*H*W, neighbor_range: int

  const auto N = pts.size(2);
  const auto C = pts_info.size(1);
  const auto B = grid_source.size(0);
  const auto H = grid_source.size(2);
  const auto W = grid_source.size(3);
  const auto NN = (2*neighbor_range+1)*(2*neighbor_range+1);

  auto options = torch::TensorOptions().dtype(pts_info.dtype()).layout(torch::kStrided).device(pts_info.device()).requires_grad(true);
  auto y = torch::zeros({1, NN, N}, options);

  // printf("x1 device: %d \n", x1.device().type()); 
  // printf("x1 index: %d \n", x1.device().index()); 

  const int threads = 1024;
  // cannot parallize across channels, because it will case modifying the the location by multiple threads at the same time
  // const dim3 blocks((n1 * n2 * channel_size + threads - 1) / threads, batch_size);
  const dim3 blocks((N  + threads - 1) / threads, NN);
  // const dim3 blocks(1, 1);

  int device_id = pts_info.device().index();
  cudaSetDevice(device_id);

  // AT_DISPATCH_FLOATING_TYPES // AT_DISPATCH_ALL_TYPES_AND_HALF
  AT_DISPATCH_FLOATING_TYPES(pts_info.type(), "cvo_dense_samp_forward_cuda", ([&] {
    cvo_dense_samp_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
      pts.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
      pts_info.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
      grid_source.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
      grid_valid.packed_accessor<bool,4,torch::RestrictPtrTraits,size_t>(),
      neighbor_range, 
      ell,
      ignore_ib,
      sqr,
      ell_basedist, 
      y.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>());
  }));
  cudaDeviceSynchronize();


  return y;
}

std::vector<torch::Tensor> cvo_dense_samp_cuda_backward(
    torch::Tensor dy, 
    torch::Tensor pts,
    torch::Tensor pts_info, 
    torch::Tensor grid_source, 
    torch::Tensor grid_valid, 
    int neighbor_range,
    float ell, 
    bool ignore_ib, 
    bool sqr,
    float ell_basedist
    ) {

  // dy: 1*NN*N

  const auto N = pts.size(2);
  const auto C = pts_info.size(1);
  const auto B = grid_source.size(0);
  const auto H = grid_source.size(2);
  const auto W = grid_source.size(3);
  const auto NN = (2*neighbor_range+1)*(2*neighbor_range+1);

  auto dx1 = torch::zeros({1, C, N}, pts_info.device());
  auto dx2 = torch::zeros({B, C, H, W}, pts_info.device());

  const int threads = 1024;

  int device_id = pts_info.device().index();
  cudaSetDevice(device_id);

  // const dim3 blocks_dx12(( N + threads - 1) / threads, C); // for cvo_dense_samp_cuda_backward_kernel_sqr_only, need y
  const dim3 blocks_dx12(( N + threads - 1) / threads);

  for (int inn = 0; inn < NN; inn++){
    // AT_DISPATCH_FLOATING_TYPES(dy.type(), "cvo_dense_samp_backward_cuda_dx", ([&] {
    //   cvo_dense_samp_cuda_backward_kernel_sqr_only<scalar_t><<<blocks_dx12, threads>>>(
    //     dx1.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
    //     dx2.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
    //     dy.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(), 
    //     y.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(), 
    //     pts.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
    //     pts_info.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
    //     grid_source.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
    //     grid_valid.packed_accessor<bool,4,torch::RestrictPtrTraits,size_t>(),
    //     neighbor_range, 
    //     ell, 
    //     ignore_ib, 
    //     sqr,
    //     ell_basedist, 
    //     inn);
    // }));
    AT_DISPATCH_FLOATING_TYPES(dy.type(), "cvo_dense_samp_backward_cuda", ([&] {
      cvo_dense_samp_cuda_backward_kernel<scalar_t><<<blocks_dx12, threads>>>(
        dx1.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        dx2.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        dy.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(), 
        pts.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        pts_info.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        grid_source.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        grid_valid.packed_accessor<bool,4,torch::RestrictPtrTraits,size_t>(),
        neighbor_range, 
        ell, 
        ignore_ib, 
        sqr,
        ell_basedist, 
        inn);
    }));
    cudaDeviceSynchronize();  
  }

  return {dx1, dx2};
}
