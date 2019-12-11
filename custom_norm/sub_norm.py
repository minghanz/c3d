import math
from torch import nn
from torch.autograd import Function
import torch

import sub_norm_cuda

class SubNormFunction(Function):
    @staticmethod
    def forward(ctx, x1, x2):
        outputs = sub_norm_cuda.forward(x1, x2)
        ctx.save_for_backward(x1, x2)
        return outputs

    @staticmethod
    def backward(ctx, dy):
        x1, x2 = ctx.saved_tensors
        dx1, dx2 = sub_norm_cuda.backward(dy, x1, x2)
        return dx1, dx2
