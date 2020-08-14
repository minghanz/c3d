#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

torch::Tensor sub_cuda_forward(
    torch::Tensor x1,
    torch::Tensor x2);

std::vector<torch::Tensor> sub_cuda_backward(
    torch::Tensor y);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor sub_forward(
    torch::Tensor x1, torch::Tensor x2) {
  CHECK_INPUT(x1);
  CHECK_INPUT(x2);

  return sub_cuda_forward(x1, x2);
}

std::vector<torch::Tensor> sub_backward(
    torch::Tensor y) {
  CHECK_INPUT(y);

  return sub_cuda_backward(y);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sub_forward, "Substraction forward (CUDA)");
  m.def("backward", &sub_backward, "Substraction backward (CUDA)");
}
