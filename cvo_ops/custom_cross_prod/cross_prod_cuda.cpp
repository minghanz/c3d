#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

torch::Tensor cross_prod_cuda_forward(
    torch::Tensor x1,
    torch::Tensor x2);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor cross_prod_forward(
    torch::Tensor x1, torch::Tensor x2) {
  CHECK_INPUT(x1);
  CHECK_INPUT(x2);

  return cross_prod_cuda_forward(x1, x2);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &cross_prod_forward, "CrossProduct forward (CUDA)");
}
