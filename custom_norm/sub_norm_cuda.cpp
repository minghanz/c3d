#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

torch::Tensor sub_norm_cuda_forward(
    torch::Tensor x1,
    torch::Tensor x2);

std::vector<torch::Tensor> sub_norm_cuda_backward(
    torch::Tensor y, 
    torch::Tensor x1,
    torch::Tensor x2);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor sub_norm_forward(
    torch::Tensor x1, torch::Tensor x2) {
  CHECK_INPUT(x1);
  CHECK_INPUT(x2);

  return sub_norm_cuda_forward(x1, x2);
}

std::vector<torch::Tensor> sub_norm_backward(
    torch::Tensor y, torch::Tensor x1, torch::Tensor x2) {
  CHECK_INPUT(y);
  CHECK_INPUT(x1);
  CHECK_INPUT(x2);

  return sub_norm_cuda_backward(y, x1, x2);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sub_norm_forward, "Substraction_norm forward (CUDA)");
  m.def("backward", &sub_norm_backward, "Substraction_norm backward (CUDA)");
}
