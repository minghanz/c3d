#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

torch::Tensor cvo_dense_angle_cuda_forward(
    torch::Tensor pts,
    torch::Tensor pts_info, 
    torch::Tensor grid_source, 
    torch::Tensor grid_valid, 
    int neighbor_range, 
    bool ignore_ib
    );

std::vector<torch::Tensor> cvo_dense_angle_cuda_backward(
    torch::Tensor dy, 
    torch::Tensor pts,
    torch::Tensor pts_info, 
    torch::Tensor grid_source, 
    torch::Tensor grid_valid, 
    int neighbor_range, 
    bool ignore_ib);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor cvo_dense_angle_forward(
    torch::Tensor pts,
    torch::Tensor pts_info, 
    torch::Tensor grid_source, 
    torch::Tensor grid_valid, 
    int neighbor_range, 
    bool ignore_ib) {
  CHECK_INPUT(pts);
  CHECK_INPUT(pts_info);
  CHECK_INPUT(grid_source);
  CHECK_INPUT(grid_valid);

  return cvo_dense_angle_cuda_forward(pts, pts_info, grid_source, grid_valid, neighbor_range, ignore_ib);
}

std::vector<torch::Tensor> cvo_dense_angle_backward(
    torch::Tensor dy, 
    torch::Tensor pts,
    torch::Tensor pts_info, 
    torch::Tensor grid_source, 
    torch::Tensor grid_valid, 
    int neighbor_range, 
    bool ignore_ib) {
  CHECK_INPUT(dy);
  CHECK_INPUT(pts);
  CHECK_INPUT(pts_info);
  CHECK_INPUT(grid_source);
  CHECK_INPUT(grid_valid);

  return cvo_dense_angle_cuda_backward(dy, pts, pts_info, grid_source, grid_valid, neighbor_range, ignore_ib);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &cvo_dense_angle_forward, "cvo_dense_angle forward (CUDA)");
  m.def("backward", &cvo_dense_angle_backward, "cvo_dense_angle backward (CUDA)");
}
