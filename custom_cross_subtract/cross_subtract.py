import math
from torch import nn
from torch.autograd import Function
import torch

import cross_subtract_cuda

class CrossSubtractFunction(Function):
    @staticmethod
    def forward(ctx, x1, x2):
        outputs = cross_subtract_cuda.forward(x1, x2)
        ctx.save_for_backward(x1, x2)
        return outputs
