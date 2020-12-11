#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <math.h>

namespace {

template <typename scalar_t>
__global__ void cvo_dense_Sigma_grid_cuda_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> pts,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> pts_info,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> grid_ells,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> grid_source,
    const torch::PackedTensorAccessor<bool,4,torch::RestrictPtrTraits,size_t> grid_valid,
    const int neighbor_range, 
    const bool ignore_ib, 
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

        float sigma_x, sigma_y, sigma_z;
        float rho_xy, rho_yz, rho_zx;
        float weight;
        float det;
        float inv_xx, inv_xy, inv_yy;
        if (C == 2){
          sigma_x = grid_ells[ib][0][v+innh][u+innw];
          sigma_y = grid_ells[ib][1][v+innh][u+innw];
          rho_xy = grid_ells[ib][2][v+innh][u+innw];
          weight = grid_ells[ib][3][v+innh][u+innw];
          det = (1 - rho_xy*rho_xy) * sigma_x * sigma_x * sigma_y * sigma_y;
          if (det < 1e-7){
            det = 1e-7;
          }
          inv_xx = sigma_y * sigma_y / det;
          inv_yy = sigma_x * sigma_x / det;
          inv_xy = -rho_xy * sigma_x * sigma_y / det;
        }
        else if (C == 3){
          sigma_x = grid_ells[ib][0][v+innh][u+innw];
          sigma_y = grid_ells[ib][1][v+innh][u+innw];
          sigma_z = grid_ells[ib][2][v+innh][u+innw];
          rho_xy = grid_ells[ib][3][v+innh][u+innw];
          rho_yz = grid_ells[ib][4][v+innh][u+innw];
          rho_zx = grid_ells[ib][5][v+innh][u+innw];
        }

        float det_neghalf = 1 / sqrt(det);
        float dx, dy, dz;
        dx = pts_info[0][0][in] - grid_source[ib][0][v+innh][u+innw];
        dy = pts_info[0][1][in] - grid_source[ib][1][v+innh][u+innw];
        if (C == 3){
          dz = pts_info[0][2][in] - grid_source[ib][2][v+innh][u+innw];
        }
        y[0][blockIdx.y][in] = det_neghalf * exp(-0.5 * (dx * dx * inv_xx + dy * dy * inv_yy + 2 * dx * dy * inv_xy)) * weight;
        
      }
    }
  }

}

template <typename scalar_t>
__global__ void cvo_dense_Sigma_grid_cuda_backward_kernel(
  torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> dx1,
  torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> dx2,
  torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> dgrid_ells,
  const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> dy,
  const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> pts,
  const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> pts_info,
  const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> grid_ells,
  const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> grid_source,
  const torch::PackedTensorAccessor<bool,4,torch::RestrictPtrTraits,size_t> grid_valid,
  const int neighbor_range, 
  const bool ignore_ib, 
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

          float sigma_x, sigma_y, sigma_z;
          float rho_xy, rho_yz, rho_zx;
          float weight;
          float det;
          float inv_xx, inv_xy, inv_yy;
          if (C == 2){
            sigma_x = grid_ells[ib][0][v+innh][u+innw];
            sigma_y = grid_ells[ib][1][v+innh][u+innw];
            rho_xy = grid_ells[ib][2][v+innh][u+innw];
            weight = grid_ells[ib][3][v+innh][u+innw];
            det = (1 - rho_xy*rho_xy) * sigma_x * sigma_x * sigma_y * sigma_y;
            if (det < 1e-7){
              det = 1e-7;
            }
            inv_xx = sigma_y * sigma_y / det;
            inv_yy = sigma_x * sigma_x / det;
            inv_xy = -rho_xy * sigma_x * sigma_y / det;
          }
          else if (C == 3){
            sigma_x = grid_ells[ib][0][v+innh][u+innw];
            sigma_y = grid_ells[ib][1][v+innh][u+innw];
            sigma_z = grid_ells[ib][2][v+innh][u+innw];
            rho_xy = grid_ells[ib][3][v+innh][u+innw];
            rho_yz = grid_ells[ib][4][v+innh][u+innw];
            rho_zx = grid_ells[ib][5][v+innh][u+innw];
          }

          float det_neghalf = 1 / sqrt(det);
          float d_x, d_y, d_z;
          d_x = pts_info[0][0][in] - grid_source[ib][0][v+innh][u+innw];
          d_y = pts_info[0][1][in] - grid_source[ib][1][v+innh][u+innw];
          if (C == 3){
            d_z = pts_info[0][2][in] - grid_source[ib][2][v+innh][u+innw];
          }
          float res_exp = exp(-0.5 * (d_x * d_x * inv_xx + d_y * d_y * inv_yy + 2 * d_x * d_y * inv_xy) );
          float res = det_neghalf * res_exp * weight;

          float res_dx_pts = - res * ( inv_xx * d_x + inv_xy * d_y );
          float res_dy_pts = - res * ( inv_xy * d_x + inv_yy * d_y );
          float res_dx_grid = - res_dx_pts;
          float res_dy_grid = - res_dy_pts;
          
          dx1[0][0][in] += dy[0][inn][in] * res_dx_pts;
          dx1[0][1][in] += dy[0][inn][in] * res_dy_pts;

          dx2[ib][0][v+innh][u+innw] += dy[0][inn][in] * res_dx_grid;
          dx2[ib][1][v+innh][u+innw] += dy[0][inn][in] * res_dy_grid;

          float ddet_neghalf_ddet = -0.5 / det / sqrt(det);
          float ddet_dsigma_x = 2 * (1 - rho_xy*rho_xy) * sigma_x * sigma_y * sigma_y;
          float ddet_dsigma_y = 2 * (1 - rho_xy*rho_xy) * sigma_x * sigma_x * sigma_y;
          float ddet_drho_xy = - 2 * rho_xy * sigma_x * sigma_x * sigma_y * sigma_y;
          
          float res_dinv_xx = res * (-0.5 * d_x * d_x);
          float res_dinv_yy = res * (-0.5 * d_y * d_y);
          float res_dinv_xy = res * (- d_x * d_y);
          float dinv_xx_dsigma_x = - inv_xx / det * ddet_dsigma_x;
          float dinv_xx_dsigma_y = (2 * sigma_y * det - sigma_y * sigma_y * ddet_dsigma_y ) / (det*det);
          float dinv_xx_drho_xy = - inv_xx / det * ddet_drho_xy;
          float dinv_yy_dsigma_x = (2 * sigma_x * det - sigma_x * sigma_x * ddet_dsigma_x) / (det*det);
          float dinv_yy_dsigma_y = - inv_yy / det * ddet_dsigma_y;
          float dinv_yy_drho_xy = - inv_yy / det * ddet_drho_xy;
          float dinv_xy_dsigma_x = (- rho_xy * sigma_y * det - rho_xy * sigma_x * sigma_y * ddet_dsigma_x) / (det*det);
          float dinv_xy_dsigma_y = (- rho_xy * sigma_x * det - rho_xy * sigma_x * sigma_y * ddet_dsigma_y) / (det*det);
          float dinv_xy_drho_xy = (- sigma_x * sigma_y * det - rho_xy * sigma_x * sigma_y * ddet_drho_xy) / (det*det);

          float res_dsigma_x = res_exp * weight * ddet_neghalf_ddet * ddet_dsigma_x + \
                                res_dinv_xx * dinv_xx_dsigma_x + res_dinv_yy * dinv_yy_dsigma_x + res_dinv_xy * dinv_xy_dsigma_x;
          
          float res_dsigma_y = res_exp * weight * ddet_neghalf_ddet * ddet_dsigma_y + \
                                res_dinv_xx * dinv_xx_dsigma_y + res_dinv_yy * dinv_yy_dsigma_y + res_dinv_xy * dinv_xy_dsigma_y;

          float res_drho_xy = res_exp * weight * ddet_neghalf_ddet * ddet_drho_xy + \
                                res_dinv_xx * dinv_xx_drho_xy + res_dinv_yy * dinv_yy_drho_xy + res_dinv_xy * dinv_xy_drho_xy;

          float res_dweight = det_neghalf * res_exp;

          dgrid_ells[ib][0][v+innh][u+innw] += dy[0][inn][in] * res_dsigma_x;
          dgrid_ells[ib][1][v+innh][u+innw] += dy[0][inn][in] * res_dsigma_y;
          dgrid_ells[ib][2][v+innh][u+innw] += dy[0][inn][in] * res_drho_xy;
          dgrid_ells[ib][3][v+innh][u+innw] += dy[0][inn][in] * res_dweight;

        }
      }
    }
  }
  
}

} // namespace

torch::Tensor cvo_dense_Sigma_grid_cuda_forward(
    torch::Tensor pts,
    torch::Tensor pts_info, 
    torch::Tensor grid_ells, 
    torch::Tensor grid_source, 
    torch::Tensor grid_valid, 
    int neighbor_range,
    bool ignore_ib
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
  AT_DISPATCH_FLOATING_TYPES(pts_info.type(), "cvo_dense_Sigma_grid_forward_cuda", ([&] {
    cvo_dense_Sigma_grid_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
      pts.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
      pts_info.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
      grid_ells.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
      grid_source.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
      grid_valid.packed_accessor<bool,4,torch::RestrictPtrTraits,size_t>(),
      neighbor_range, 
      ignore_ib,
      y.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>());
  }));
  cudaDeviceSynchronize();


  return y;
}

std::vector<torch::Tensor> cvo_dense_Sigma_grid_cuda_backward(
    torch::Tensor dy, 
    torch::Tensor pts,
    torch::Tensor pts_info, 
    torch::Tensor grid_ells, 
    torch::Tensor grid_source, 
    torch::Tensor grid_valid, 
    int neighbor_range,
    bool ignore_ib
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
  auto dgrid_ells = torch::zeros({B, 4, H, W}, pts_info.device());  // for C==2, 3 corresponds to sigma_x, sigma_y, rho_xy

  const int threads = 1024;

  int device_id = pts_info.device().index();
  cudaSetDevice(device_id);

  // const dim3 blocks_dx12(( N + threads - 1) / threads, C); // for cvo_dense_Sigma_cuda_backward_kernel_sqr_only, need y
  const dim3 blocks_dx12(( N + threads - 1) / threads);

  for (int inn = 0; inn < NN; inn++){
    AT_DISPATCH_FLOATING_TYPES(dy.type(), "cvo_dense_Sigma_grid_backward_cuda", ([&] {
      cvo_dense_Sigma_grid_cuda_backward_kernel<scalar_t><<<blocks_dx12, threads>>>(
        dx1.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        dx2.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        dgrid_ells.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        dy.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(), 
        pts.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        pts_info.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        grid_ells.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        grid_source.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        grid_valid.packed_accessor<bool,4,torch::RestrictPtrTraits,size_t>(),
        neighbor_range, 
        ignore_ib, 
        inn);
    }));
    cudaDeviceSynchronize();  
  }

  return {dx1, dx2, dgrid_ells};
}
