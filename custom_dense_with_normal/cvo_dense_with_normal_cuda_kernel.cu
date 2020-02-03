#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <math.h>

namespace {

template <typename scalar_t>
__global__ void cvo_dense_with_normal_cuda_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> pts,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> pts_info,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> grid_source,
    const torch::PackedTensorAccessor<bool,4,torch::RestrictPtrTraits,size_t> grid_valid,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> pts_normal,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> grid_normal,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> pts_nres,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> grid_nres,
    const int neighbor_range, 
    const float ell,
    const float mag_max,
    const float mag_min,
    const bool ignore_ib, 
    const bool norm_in_dist, 
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

        if (norm_in_dist){ // TODO: this part is not ready yet
          // float dx = 0;
          // float dx_n_pts = 0;
          // float dx_n_grid = 0;
          // float dx_c = 0;
          // for (int ic = 0; ic < C; ic++){
          //   dx_c = pts_info[0][ic][in] - grid_source[ib][ic][v+innh][u+innw];
          //   dx += dx_c * dx_c;
          //   dx_n_pts += dx_c * pts_normal[0][ic][in];
          //   dx_n_grid += dx_c * grid_normal[ib][ic][v+innh][u+innw];
          //   // y[0][blockIdx.y][in] += (pts_info[0][ic][in] - grid_source[ib][ic][v+innh][u+innw]) * (pts_info[0][ic][in] - grid_source[ib][ic][v+innh][u+innw]);
          // }
          // dx_n_pts = dx_n_pts * dx_n_pts;
          // dx_n_grid = dx_n_grid * dx_n_grid;
          // float alpha = pts_nres[0][0][in];
          // float d = dx_n_pts + alpha * (dx - dx_n_pts);
          // y[0][blockIdx.y][in] = 1/alpha * exp(- sqrt(d)/ell);
          // // y[0][blockIdx.y][in] =  exp( - y[0][blockIdx.y][in] / (2*ell*ell) ) ;

          float ntn = 0;
          float dx_n_pts = 0;
          float dx_n_grid = 0;
          for (int ic = 0; ic < C; ic++){
            ntn += pts_normal[0][ic][in] * grid_normal[ib][ic][v+innh][u+innw];
            dx_n_pts += (pts_info[0][ic][in] - grid_source[ib][ic][v+innh][u+innw]) * pts_normal[0][ic][in];
            dx_n_grid += (pts_info[0][ic][in] - grid_source[ib][ic][v+innh][u+innw]) * grid_normal[ib][ic][v+innh][u+innw];
          }
          float res = pts_nres[0][0][in] + grid_nres[ib][0][v+innh][u+innw];
          float alpha = 2 * mag_min / (2*mag_min/mag_max + res);
          float dx_n = max(fabs(dx_n_grid), fabs(dx_n_pts));
          y[0][blockIdx.y][in] = (fabs(ntn)+1e-8) * alpha * exp(- dx_n/ell_apply);
        }
        else{
          float dx = 0;
          float ntn = 0;
          for (int ic = 0; ic < C; ic++){
            dx += (pts_info[0][ic][in] - grid_source[ib][ic][v+innh][u+innw]) * (pts_info[0][ic][in] - grid_source[ib][ic][v+innh][u+innw]);
            ntn += pts_normal[0][ic][in] * grid_normal[ib][ic][v+innh][u+innw];
          }
          float res = pts_nres[0][0][in] + grid_nres[ib][0][v+innh][u+innw];
          float alpha = 2 * mag_min / (2*mag_min/mag_max + res);
          y[0][blockIdx.y][in] = (fabs(ntn)+1e-8) * alpha * exp(- sqrt(dx+1e-8)/ell_apply);

        }
        
      }
    }
  }

}


template <typename scalar_t>
__global__ void cvo_dense_with_normal_cuda_backward_kernel_dx(
  torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> dx1,
  torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> dx2,
  torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> dn1,
  torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> dn2,
  torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> dr1,
  torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> dr2,
  const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> dy, 
  const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> pts,
  const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> pts_info,
  const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> grid_source,
  const torch::PackedTensorAccessor<bool,4,torch::RestrictPtrTraits,size_t> grid_valid,
  const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> pts_normal,
  const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> grid_normal,
  const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> pts_nres,
  const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> grid_nres,
  const int neighbor_range, 
  const float ell, 
  const float mag_max,
  const float mag_min,
  const bool ignore_ib, 
  const bool norm_in_dist, 
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

          if (norm_in_dist){

            float dx_n_pts = 0;
            float dx_n_grid = 0;
            float ntn = 0;
            for (int ic = 0; ic < C; ic++){
              dx_n_pts += (pts_info[0][ic][in] - grid_source[ib][ic][v+innh][u+innw]) * pts_normal[0][ic][in];
              dx_n_grid += (pts_info[0][ic][in] - grid_source[ib][ic][v+innh][u+innw]) * grid_normal[ib][ic][v+innh][u+innw];
              ntn += pts_normal[0][ic][in] * grid_normal[ib][ic][v+innh][u+innw];
            }
            bool neg_ntn = ntn < 0;
            float sign_ntn = 1;
            if (neg_ntn){
              ntn = -ntn;
              sign_ntn = -1;
            }
            bool neg_n_pts = dx_n_pts < 0;
            float sign_n_pts = 1;
            if (neg_n_pts){
              dx_n_pts = -dx_n_pts;
              sign_n_pts = -1;
            }
            bool neg_n_grid = dx_n_grid < 0;
            float sign_n_grid = 1;
            if (neg_n_grid){
              dx_n_grid = -dx_n_grid;
              sign_n_grid = -1;
            }
            bool max_at_pts;
            float dx_n;
            if (dx_n_pts > dx_n_grid){
              max_at_pts = true;
              dx_n = dx_n_pts;
            }
            else{
              max_at_pts = false;
              dx_n = dx_n_grid;
            }
            float res = pts_nres[0][0][in] + grid_nres[ib][0][v+innh][u+innw];
            float alpha = 2 * mag_min / (2*mag_min/mag_max + res);
            float y_cur = (ntn+1e-8) * alpha * exp(- dx_n/ell_apply);
            for (int ic = 0; ic < C; ic++){
              if (max_at_pts){
                dx1[0][ic][in] -= dy[0][inn][in] * y_cur * pts_normal[0][ic][in] / ell_apply * sign_n_pts ;
                dx2[ib][ic][v+innh][u+innw] += dy[0][inn][in] * y_cur * pts_normal[0][ic][in] / ell_apply * sign_n_pts ;
              }
              else{
                dx1[0][ic][in] -= dy[0][inn][in] * y_cur * grid_normal[ib][ic][v+innh][u+innw] / ell_apply * sign_n_grid ;
                dx2[ib][ic][v+innh][u+innw] += dy[0][inn][in] * y_cur * grid_normal[ib][ic][v+innh][u+innw] / ell_apply * sign_n_grid ;
              }
              dn1[0][ic][in] += dy[0][inn][in] * sign_ntn * grid_normal[ib][ic][v+innh][u+innw] * y_cur / (ntn+1e-8);
              dn2[ib][ic][v+innh][u+innw] += dy[0][inn][in] * sign_ntn * pts_normal[0][ic][in] * y_cur / (ntn+1e-8);
              dr1[0][0][in] -= dy[0][inn][in] * y_cur / (2*mag_min/mag_max + res);
              dr2[ib][0][v+innh][u+innw] -= dy[0][inn][in] * y_cur / (2*mag_min/mag_max + res);
            }
          }
          else{
            float dx = 0;
            float ntn = 0;
            for (int ic = 0; ic < C; ic++){
              dx += (pts_info[0][ic][in] - grid_source[ib][ic][v+innh][u+innw]) * (pts_info[0][ic][in] - grid_source[ib][ic][v+innh][u+innw]);
              ntn += pts_normal[0][ic][in] * grid_normal[ib][ic][v+innh][u+innw];
            }
            dx = sqrt(dx+1e-8);
            bool neg_ntn = ntn < 0;
            float sign_ntn = 1;
            if (neg_ntn){
              ntn = -ntn;
              sign_ntn = -1;
            }
            float res = pts_nres[0][0][in] + grid_nres[ib][0][v+innh][u+innw];
            float alpha = 2 * mag_min / (2*mag_min/mag_max + res);
            float y_cur = (ntn+1e-8) * alpha * exp(- dx/ell_apply);
            for (int ic = 0; ic < C; ic++){
              dx1[0][ic][in] += dy[0][inn][in] * y_cur * (grid_source[ib][ic][v+innh][u+innw] - pts_info[0][ic][in]) / ell_apply / dx;
              dx2[ib][ic][v+innh][u+innw] -= dy[0][inn][in] * y_cur * (grid_source[ib][ic][v+innh][u+innw] - pts_info[0][ic][in]) / ell_apply / dx;
              dn1[0][ic][in] += dy[0][inn][in] * sign_ntn * grid_normal[ib][ic][v+innh][u+innw] * y_cur / (ntn+1e-8);
              dn2[ib][ic][v+innh][u+innw] += dy[0][inn][in] * sign_ntn * pts_normal[0][ic][in] * y_cur / (ntn+1e-8);
              dr1[0][0][in] -= dy[0][inn][in] * y_cur / (2*mag_min/mag_max + res);
              dr2[ib][0][v+innh][u+innw] -= dy[0][inn][in] * y_cur / (2*mag_min/mag_max + res);
            }
          }
          // dx1[0][blockIdx.y][in] += dy[0][inn][in] * y[0][inn][in] * (grid_source[ib][blockIdx.y][v+innh][u+innw] - pts_info[0][blockIdx.y][in]) / (ell*ell);
          // dx2[ib][blockIdx.y][v+innh][u+innw] -= dy[0][inn][in] * y[0][inn][in] * (grid_source[ib][blockIdx.y][v+innh][u+innw] - pts_info[0][blockIdx.y][in]) / (ell*ell);
        }
      }
    }
  }
  
}

} // namespace

torch::Tensor cvo_dense_with_normal_cuda_forward(
    torch::Tensor pts,
    torch::Tensor pts_info, 
    torch::Tensor grid_source, 
    torch::Tensor grid_valid, 
    torch::Tensor pts_normal,
    torch::Tensor grid_normal, 
    torch::Tensor pts_nres,
    torch::Tensor grid_nres, 
    int neighbor_range,
    float ell, 
    float mag_max,
    float mag_min,
    bool ignore_ib, 
    bool norm_in_dist,
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
  AT_DISPATCH_FLOATING_TYPES(pts_info.type(), "cvo_dense_with_normal_forward_cuda", ([&] {
    cvo_dense_with_normal_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
      pts.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
      pts_info.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
      grid_source.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
      grid_valid.packed_accessor<bool,4,torch::RestrictPtrTraits,size_t>(),
      pts_normal.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
      grid_normal.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
      pts_nres.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
      grid_nres.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
      neighbor_range, 
      ell,
      mag_max,
      mag_min,
      ignore_ib, 
      norm_in_dist, 
      ell_basedist,
      y.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>());
  }));
  cudaDeviceSynchronize();


  return y;
}

std::vector<torch::Tensor> cvo_dense_with_normal_cuda_backward(
    torch::Tensor dy, 
    torch::Tensor pts,
    torch::Tensor pts_info, 
    torch::Tensor grid_source, 
    torch::Tensor grid_valid, 
    torch::Tensor pts_normal,
    torch::Tensor grid_normal, 
    torch::Tensor pts_nres,
    torch::Tensor grid_nres, 
    int neighbor_range,
    float ell, 
    float mag_max,
    float mag_min,
    bool ignore_ib, 
    bool norm_in_dist, 
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
  auto dn1 = torch::zeros({1, C, N}, pts_info.device());
  auto dn2 = torch::zeros({B, C, H, W}, pts_info.device());
  auto dr1 = torch::zeros({1, 1, N}, pts_info.device());
  auto dr2 = torch::zeros({B, 1, H, W}, pts_info.device());

  const int threads = 512;

  int device_id = pts_info.device().index();
  cudaSetDevice(device_id);

  const dim3 blocks_dx12(( N + threads - 1) / threads);
  // const dim3 blocks_dx12(( N + threads - 1) / threads, C);

  for (int inn = 0; inn < NN; inn++){
    AT_DISPATCH_FLOATING_TYPES(dy.type(), "cvo_dense_with_normal_backward_cuda_dx", ([&] {
      cvo_dense_with_normal_cuda_backward_kernel_dx<scalar_t><<<blocks_dx12, threads>>>(
        dx1.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        dx2.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        dn1.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        dn2.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        dr1.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        dr2.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        dy.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(), 
        pts.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        pts_info.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        grid_source.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        grid_valid.packed_accessor<bool,4,torch::RestrictPtrTraits,size_t>(),
        pts_normal.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        grid_normal.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        pts_nres.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        grid_nres.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        neighbor_range, 
        ell, 
        mag_max,
        mag_min,
        ignore_ib, 
        norm_in_dist, 
        ell_basedist,
        inn);
    }));
    cudaDeviceSynchronize();  
  }

  return {dx1, dx2, dn1, dn2, dr1, dr2};
}
