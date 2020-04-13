import math
from torch import nn
from torch.autograd import Function
import torch

import sub_cuda

class SubFunction(Function):
    @staticmethod
    def forward(ctx, x1, x2):
        outputs = sub_cuda.forward(x1, x2)
        return outputs

    @staticmethod
    def backward(ctx, dy):
        dx1, dx2 = sub_cuda.backward(dy)
        return dx1, dx2
