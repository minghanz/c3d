#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> cvo_dense_with_normal_cuda_forward(
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
    float ell_basedist, 
    bool return_normal_kernel
    );

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
    float ell_basedist);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> cvo_dense_with_normal_forward(
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
    float ell_basedist, 
    bool return_normal_kernel) {
  CHECK_INPUT(pts);
  CHECK_INPUT(pts_info);
  CHECK_INPUT(grid_source);
  CHECK_INPUT(grid_valid);
  CHECK_INPUT(pts_normal);
  CHECK_INPUT(grid_normal);
  CHECK_INPUT(pts_nres);
  CHECK_INPUT(grid_nres);

  return cvo_dense_with_normal_cuda_forward(pts, pts_info, grid_source, grid_valid, pts_normal, grid_normal, pts_nres, grid_nres, neighbor_range, ell, mag_max, mag_min, ignore_ib, norm_in_dist, ell_basedist, return_normal_kernel);
}

std::vector<torch::Tensor> cvo_dense_with_normal_backward(
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
    float ell_basedist) {
  CHECK_INPUT(dy);
  CHECK_INPUT(pts);
  CHECK_INPUT(pts_info);
  CHECK_INPUT(grid_source);
  CHECK_INPUT(grid_valid);
  CHECK_INPUT(pts_normal);
  CHECK_INPUT(grid_normal);
  CHECK_INPUT(pts_nres);
  CHECK_INPUT(grid_nres);

  return cvo_dense_with_normal_cuda_backward(dy, pts, pts_info, grid_source, grid_valid, pts_normal, grid_normal, pts_nres, grid_nres, neighbor_range, ell, mag_max, mag_min, ignore_ib, norm_in_dist, ell_basedist);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &cvo_dense_with_normal_forward, "cvo_dense_with_normal forward (CUDA)");
  m.def("backward", &cvo_dense_with_normal_backward, "cvo_dense_with_normal backward (CUDA)");
}
