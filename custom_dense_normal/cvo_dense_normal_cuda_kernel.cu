#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <math.h>

namespace {

template <typename scalar_t>
__global__ void cvo_dense_normal_cuda_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> pts,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> grid_source,
    const torch::PackedTensorAccessor<bool,4,torch::RestrictPtrTraits,size_t> grid_valid,
    const int neighbor_range, 
    const bool ignore_ib, 
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> y, 
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> pnorm, 
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> ioffs) {

  const auto N = pts.size(2);
  const auto C = grid_source.size(1);
  const auto B = grid_source.size(0);
  const auto H = grid_source.size(2);
  const auto W = grid_source.size(3);

  //dim3 block[N, 4, 1]
  const auto in = blockIdx.x * blockDim.x + threadIdx.x;  
  const int ipair = blockIdx.y;

  if (in < N ){
    const int u = pts[0][0][in];  // u:right (col), v:down (row)
    const int v = pts[0][1][in];
    int ib;
    if (ignore_ib){
      ib = 0;
    }
    else{
      ib = pts[0][2][in];
    }
    int u_inc_0, v_inc_0, u_inc_1, v_inc_1;
    if (ipair == 0){
      u_inc_0 = 0;
      v_inc_0 = -1;
      u_inc_1 = 1;
      v_inc_1 = 0;
    }
    else if (ipair == 1){
      u_inc_0 = 1;
      v_inc_0 = -1;
      u_inc_1 = 1;
      v_inc_1 = 1;
    }
    else if(ipair == 2){
      u_inc_0 = 0;
      v_inc_0 = 1;
      u_inc_1 = -1;
      v_inc_1 = 0;
    }
    else if(ipair == 3){
      u_inc_0 = -1;
      v_inc_0 = 1;
      u_inc_1 = -1;
      v_inc_1 = -1;
    }
    // TODO: check to make sure i_pair < 4?
    int u_cur_0, v_cur_0, u_cur_1, v_cur_1;
    bool p0_found = false;
    bool p1_found = false;
    int ioff_0;
    int ioff_1;
    for (ioff_0 = 1; ioff_0 <= neighbor_range; ioff_0++){
      u_cur_0 = u + u_inc_0*ioff_0;
      v_cur_0 = v + v_inc_0*ioff_0;
      if (u_cur_0 >= 0 && u_cur_0 < W && v_cur_0 >= 0 && v_cur_0 < H){
        if (grid_valid[ib][0][v_cur_0][u_cur_0] > 0){
          p0_found = true;
          break;
        }
      }
    }
    for (ioff_1 = 1; ioff_1 <= neighbor_range; ioff_1++){
      u_cur_1 = u + u_inc_1*ioff_1;
      v_cur_1 = v + v_inc_1*ioff_1;
      if (u_cur_1 >= 0 && u_cur_1 < W && v_cur_1 >= 0 && v_cur_1 < H){
        if (grid_valid[ib][0][v_cur_1][u_cur_1] > 0){
          p1_found = true;
          break;
        }
      }
    }
    if (p0_found && p1_found){
      y[ipair][0][in] = (grid_source[ib][1][v_cur_0][u_cur_0] - grid_source[ib][1][v][u]) * (grid_source[ib][2][v_cur_1][u_cur_1] - grid_source[ib][2][v][u]) - 
                        (grid_source[ib][2][v_cur_0][u_cur_0] - grid_source[ib][2][v][u]) * (grid_source[ib][1][v_cur_1][u_cur_1] - grid_source[ib][1][v][u]);

      y[ipair][1][in] = (grid_source[ib][2][v_cur_0][u_cur_0] - grid_source[ib][2][v][u]) * (grid_source[ib][0][v_cur_1][u_cur_1] - grid_source[ib][0][v][u]) - 
                        (grid_source[ib][0][v_cur_0][u_cur_0] - grid_source[ib][0][v][u]) * (grid_source[ib][2][v_cur_1][u_cur_1] - grid_source[ib][2][v][u]);

      y[ipair][2][in] = (grid_source[ib][0][v_cur_0][u_cur_0] - grid_source[ib][0][v][u]) * (grid_source[ib][1][v_cur_1][u_cur_1] - grid_source[ib][1][v][u]) - 
                        (grid_source[ib][1][v_cur_0][u_cur_0] - grid_source[ib][1][v][u]) * (grid_source[ib][0][v_cur_1][u_cur_1] - grid_source[ib][0][v][u]);

      pnorm[ipair][0][in] = (grid_source[ib][0][v_cur_0][u_cur_0] - grid_source[ib][0][v][u]) * (grid_source[ib][0][v_cur_0][u_cur_0] - grid_source[ib][0][v][u]) + 
                      (grid_source[ib][1][v_cur_0][u_cur_0] - grid_source[ib][1][v][u]) * (grid_source[ib][1][v_cur_0][u_cur_0] - grid_source[ib][1][v][u]) + 
                      (grid_source[ib][2][v_cur_0][u_cur_0] - grid_source[ib][2][v][u]) * (grid_source[ib][2][v_cur_0][u_cur_0] - grid_source[ib][2][v][u]);
      pnorm[ipair][1][in] = (grid_source[ib][0][v_cur_1][u_cur_1] - grid_source[ib][0][v][u]) * (grid_source[ib][0][v_cur_1][u_cur_1] - grid_source[ib][0][v][u]) + 
                      (grid_source[ib][1][v_cur_1][u_cur_1] - grid_source[ib][1][v][u]) * (grid_source[ib][1][v_cur_1][u_cur_1] - grid_source[ib][1][v][u]) + 
                      (grid_source[ib][2][v_cur_1][u_cur_1] - grid_source[ib][2][v][u]) * (grid_source[ib][2][v_cur_1][u_cur_1] - grid_source[ib][2][v][u]);

      ioffs[ipair][0][in] = ioff_0;
      ioffs[ipair][1][in] = ioff_1;
    }
  }

}


template <typename scalar_t>
__global__ void cvo_dense_normal_cuda_backward_kernel_dx(
  torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> dgrid,
  const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> dy, 
  const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> dpnorm, 
  const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> ioffs, 
  const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> pts,
  const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> grid_source,
  const bool ignore_ib, 
  const int inn) {
  // dx1: 1*C*N
  // dx2: B*C*H*W
  // dy: 1*NN*N

  const auto N = pts.size(2);
  const auto C = grid_source.size(1);
  const auto B = grid_source.size(0);
  const auto H = grid_source.size(2);
  const auto W = grid_source.size(3);

  //dim3 block[N, C, 1] 
  if (inn < 8){
    const int in = blockIdx.x * blockDim.x + threadIdx.x;

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

      const int ipair = inn % 4;
      const int ip_self = inn / 4;
      const int ip_other = 1 - ip_self;

      const int ioff_self = ioffs[ipair][ip_self][in];
      const int ioff_other = ioffs[ipair][ip_other][in];
      if (ioff_self > 0 && ioff_other > 0){
        int u_inc_0, v_inc_0, u_inc_1, v_inc_1;
        if (ipair == 0){
          u_inc_0 = 0;
          v_inc_0 = -1;
          u_inc_1 = 1;
          v_inc_1 = 0;
        }
        else if (ipair == 1){
          u_inc_0 = 1;
          v_inc_0 = -1;
          u_inc_1 = 1;
          v_inc_1 = 1;
        }
        else if(ipair == 2){
          u_inc_0 = 0;
          v_inc_0 = 1;
          u_inc_1 = -1;
          v_inc_1 = 0;
        }
        else if(ipair == 3){
          u_inc_0 = -1;
          v_inc_0 = 1;
          u_inc_1 = -1;
          v_inc_1 = -1;
        }
        int u_cur_self, v_cur_self, u_cur_other, v_cur_other;
        float sign_dydp;
        if (ip_self == 0){
          u_cur_self = u + u_inc_0 * ioff_self;
          v_cur_self = v + v_inc_0 * ioff_self;
          u_cur_other = u + u_inc_1 * ioff_other;
          v_cur_other = v + v_inc_1 * ioff_other;
          sign_dydp = 1;
        }
        else{
          u_cur_self = u + u_inc_1 * ioff_self;
          v_cur_self = v + v_inc_1 * ioff_self;
          u_cur_other = u + u_inc_0 * ioff_other;
          v_cur_other = v + v_inc_0 * ioff_other;
          sign_dydp = -1;
        }
        float dy0 = dy[ipair][0][in];
        float dy1 = dy[ipair][1][in];
        float dy2 = dy[ipair][2][in];
        
        float ic = blockIdx.y;
        if (ic == 0){
          dgrid[ib][0][v_cur_self][u_cur_self] += sign_dydp * (-(grid_source[ib][2][v_cur_other][u_cur_other] - grid_source[ib][2][v][u]) * dy[ipair][1][in] + 
                                                                (grid_source[ib][1][v_cur_other][u_cur_other] - grid_source[ib][1][v][u]) * dy[ipair][2][in]);
        }
        else if(ic == 1){
          dgrid[ib][1][v_cur_self][u_cur_self] += sign_dydp * ((grid_source[ib][2][v_cur_other][u_cur_other] - grid_source[ib][2][v][u]) * dy[ipair][0][in] - 
                                                                (grid_source[ib][0][v_cur_other][u_cur_other] - grid_source[ib][0][v][u]) * dy[ipair][2][in]);
        }
        else if(ic == 2){
          dgrid[ib][2][v_cur_self][u_cur_self] += sign_dydp * (-(grid_source[ib][1][v_cur_other][u_cur_other] - grid_source[ib][1][v][u]) * dy[ipair][0][in] + 
                                                                (grid_source[ib][0][v_cur_other][u_cur_other] - grid_source[ib][0][v][u]) * dy[ipair][1][in]);
        }
        dgrid[ib][ic][v_cur_self][u_cur_self] += 2 * (grid_source[ib][ic][v_cur_self][u_cur_self] - grid_source[ib][ic][v][u]) * dpnorm[ipair][ip_self][in];

        
        // dgrid[ib][0][v_cur_self][u_cur_self] += sign_dydp * (-(grid_source[ib][2][v_cur_other][u_cur_other] - grid_source[ib][2][v][u]) * dy1 + (grid_source[ib][1][v_cur_other][u_cur_other] - grid_source[ib][1][v][u]) * dy2)
        // dgrid[ib][1][v_cur_self][u_cur_self] += sign_dydp * ((grid_source[ib][2][v_cur_other][u_cur_other] - grid_source[ib][2][v][u]) * dy0 - (grid_source[ib][0][v_cur_other][u_cur_other] - grid_source[ib][0][v][u]) * dy2)
        // dgrid[ib][2][v_cur_self][u_cur_self] += sign_dydp * (-(grid_source[ib][1][v_cur_other][u_cur_other] - grid_source[ib][1][v][u]) * dy0 + (grid_source[ib][0][v_cur_other][u_cur_other] - grid_source[ib][0][v][u]) * dy1)

        // dgrid[ib][0][v_cur_self][u_cur_self] += 2 * (grid_source[ib][0][v_cur_self][u_cur_self] - grid_source[ib][0][v][u]) * dpnorm[ipair][ip_self][in]

        // dydp0x = -(grid_source[ib][2][v_cur_1][u_cur_1] - grid_source[ib][2][v][u]) * dy1 + (grid_source[ib][1][v_cur_1][u_cur_1] - grid_source[ib][1][v][u]) * dy2
        // dydpx -= dydp0x
        // dydp0y = (grid_source[ib][2][v_cur_1][u_cur_1] - grid_source[ib][2][v][u]) * dy0 - (grid_source[ib][0][v_cur_1][u_cur_1] - grid_source[ib][0][v][u]) * dy2
        // dydpy -= dydp0y
        // dydp0z = -(grid_source[ib][1][v_cur_1][u_cur_1] - grid_source[ib][1][v][u]) * dy0 + (grid_source[ib][0][v_cur_1][u_cur_1] - grid_source[ib][0][v][u]) * dy1
        // dydpz -= dydp0z
        // dydp1x = (grid_source[ib][2][v_cur_0][u_cur_0] - grid_source[ib][2][v][u]) * dy1 - (grid_source[ib][1][v_cur_0][u_cur_0] - grid_source[ib][1][v][u]) * dy2
        // dydpx -= dydp1x
        // dydp1y = -(grid_source[ib][2][v_cur_0][u_cur_0] - grid_source[ib][2][v][u]) * dy0 + (grid_source[ib][0][v_cur_0][u_cur_0] - grid_source[ib][0][v][u]) * dy2
        // dydpy -= dydp1y
        // dydp1z = (grid_source[ib][1][v_cur_0][u_cur_0] - grid_source[ib][1][v][u]) * dy0 - (grid_source[ib][0][v_cur_0][u_cur_0] - grid_source[ib][0][v][u]) * dy1
        // dydpz -= dydp1z

        // dp0x = 2 * (grid_source[ib][0][v_cur_0][u_cur_0] - grid_source[ib][0][v][u]) * dpn0
        // dpx -= dp0x
        // dp0y = 2 * (grid_source[ib][1][v_cur_0][u_cur_0] - grid_source[ib][1][v][u]) * dpn0
        // dpy -= dp0y
        // dp0z = 2 * (grid_source[ib][2][v_cur_0][u_cur_0] - grid_source[ib][2][v][u]) * dpn0
        // dpz -= dp0z

        // dp1x = 2 * (grid_source[ib][0][v_cur_1][u_cur_1] - grid_source[ib][0][v][u]) * dpn1
        // dpx -= dp1x
        // dp1y = 2 * (grid_source[ib][1][v_cur_1][u_cur_1] - grid_source[ib][1][v][u]) * dpn1
        // dpy -= dp1y
        // dp1z = 2 * (grid_source[ib][2][v_cur_1][u_cur_1] - grid_source[ib][2][v][u]) * dpn1
        // dpz -= dp1z
        
        
      }
      
    }
  }
  
}

template <typename scalar_t>
__global__ void cvo_dense_normal_cuda_backward_kernel_dx_m(
  const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> dgrid,
  torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> dgrid_m,
  const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> ioffs, 
  const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> pts,
  const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> grid_source,
  const bool ignore_ib) {

  const auto N = pts.size(2);
  const auto C = grid_source.size(1);
  const auto B = grid_source.size(0);
  const auto H = grid_source.size(2);
  const auto W = grid_source.size(3);

  const int in = blockIdx.x * blockDim.x + threadIdx.x;

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
    for (int ipair = 0; ipair < 4; ipair++){

      const int ioff_0 = ioffs[ipair][0][in];
      const int ioff_1 = ioffs[ipair][1][in];

      if (ioff_0 > 0 && ioff_1 > 0){
        int u_inc_0, v_inc_0, u_inc_1, v_inc_1;
        if (ipair == 0){
          u_inc_0 = 0;
          v_inc_0 = -1;
          u_inc_1 = 1;
          v_inc_1 = 0;
        }
        else if (ipair == 1){
          u_inc_0 = 1;
          v_inc_0 = -1;
          u_inc_1 = 1;
          v_inc_1 = 1;
        }
        else if(ipair == 2){
          u_inc_0 = 0;
          v_inc_0 = 1;
          u_inc_1 = -1;
          v_inc_1 = 0;
        }
        else if(ipair == 3){
          u_inc_0 = -1;
          v_inc_0 = 1;
          u_inc_1 = -1;
          v_inc_1 = -1;
        }
        int u_cur_0 = u + u_inc_0 * ioff_0;
        int v_cur_0 = v + v_inc_0 * ioff_0;
        int u_cur_1 = u + u_inc_1 * ioff_1;
        int v_cur_1 = v + v_inc_1 * ioff_1;

        dgrid_m[ib][blockIdx.y][v][u] -= dgrid[ib][blockIdx.y][v_cur_0][u_cur_0];
        dgrid_m[ib][blockIdx.y][v][u] -= dgrid[ib][blockIdx.y][v_cur_1][u_cur_1];
      }
      
    }
    
  }

}

} // namespace

std::vector<torch::Tensor> cvo_dense_normal_cuda_forward(
    torch::Tensor pts,
    torch::Tensor grid_source, 
    torch::Tensor grid_valid, 
    int neighbor_range,
    bool ignore_ib
    ) {
    // pts: 1*2*N, grid_source: B*C*H*W (C could be xyz, rgb, ...), 
    // grid_valid: B*1*H*W, neighbor_range: int

  const auto N = pts.size(2);
  const auto C = grid_source.size(1);
  const auto B = grid_source.size(0);
  const auto H = grid_source.size(2);
  const auto W = grid_source.size(3);

  auto options = torch::TensorOptions().dtype(grid_source.dtype()).layout(torch::kStrided).device(grid_source.device()).requires_grad(true);
  auto y = torch::zeros({4, C, N}, options);
  auto pnorm = torch::zeros({4, 2, N}, options);
  auto ioffs = torch::zeros({4, 2, N}, options);

  // printf("x1 device: %d \n", x1.device().type()); 
  // printf("x1 index: %d \n", x1.device().index()); 

  const int threads = 1024;
  // cannot parallize across channels, because it will case modifying the the location by multiple threads at the same time
  // const dim3 blocks((n1 * n2 * channel_size + threads - 1) / threads, batch_size);
  const dim3 blocks((N  + threads - 1) / threads, 4);
  // const dim3 blocks(1, 1);

  int device_id = grid_source.device().index();
  cudaSetDevice(device_id);

  // AT_DISPATCH_FLOATING_TYPES // AT_DISPATCH_ALL_TYPES_AND_HALF
  AT_DISPATCH_FLOATING_TYPES(grid_source.type(), "cvo_dense_normal_forward_cuda", ([&] {
    cvo_dense_normal_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
      pts.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
      grid_source.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
      grid_valid.packed_accessor<bool,4,torch::RestrictPtrTraits,size_t>(),
      neighbor_range, 
      ignore_ib, 
      y.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
      pnorm.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
      ioffs.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>());
  }));
  cudaDeviceSynchronize();


  return {y, pnorm, ioffs};
}

torch::Tensor cvo_dense_normal_cuda_backward(
    torch::Tensor dy, 
    torch::Tensor dpnorm, 
    torch::Tensor ioffs, 
    torch::Tensor pts,
    torch::Tensor grid_source, 
    bool ignore_ib
    ) {

  // dy: 1*NN*N

  const auto N = pts.size(2);
  const auto C = grid_source.size(1);
  const auto B = grid_source.size(0);
  const auto H = grid_source.size(2);
  const auto W = grid_source.size(3);

  auto dgrid = torch::zeros({B, C, H, W}, grid_source.device());

  int device_id = grid_source.device().index();
  cudaSetDevice(device_id);

  const int threads = 1024;
  const dim3 blocks_dx12(( N + threads - 1) / threads, C);

  for (int inn = 0; inn < 8; inn++){
    AT_DISPATCH_FLOATING_TYPES(dy.type(), "cvo_dense_normal_backward_cuda_dx", ([&] {
      cvo_dense_normal_cuda_backward_kernel_dx<scalar_t><<<blocks_dx12, threads>>>(
        dgrid.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        dy.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(), 
        dpnorm.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(), 
        ioffs.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(), 
        pts.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        grid_source.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        ignore_ib, 
        inn);
    }));
    cudaDeviceSynchronize();  
  }

  auto dgrid_m = torch::zeros({B, C, H, W}, grid_source.device());

  AT_DISPATCH_FLOATING_TYPES(dy.type(), "cvo_dense_normal_backward_cuda_dx_m", ([&] {
    cvo_dense_normal_cuda_backward_kernel_dx_m<scalar_t><<<blocks_dx12, threads>>>(
      dgrid.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
      dgrid_m.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
      ioffs.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(), 
      pts.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
      grid_source.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
      ignore_ib);
  }));
  cudaDeviceSynchronize();  

  dgrid = dgrid + dgrid_m;
  return dgrid;
}
