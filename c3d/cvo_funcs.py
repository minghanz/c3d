import torch
from torch.autograd import Function
from .cvo_ops import *
from .utils.io import save_nkern

'''
from pytorch-unet/geometry.py
'''
# import sub_cuda
# import cross_prod_cuda, cross_subtract_cuda, sub_norm_cuda_half_paral #sub_norm_cuda sub_norm_cuda_half 

class SubNormFunction(Function):
    '''square of L2 norm''' 
    @staticmethod
    def forward(ctx, x1, x2):
        outputs = sub_norm_cuda_half_paral.forward(x1, x2)
        ctx.save_for_backward(x1, x2)
        return outputs

    @staticmethod
    def backward(ctx, dy):
        x1, x2 = ctx.saved_tensors

        dx1, dx2 = sub_norm_cuda_half_paral.backward(dy, x1, x2)
        return dx1, dx2

class CrossProdFunction(Function):
    '''cross product'''
    @staticmethod
    def forward(ctx, x1, x2):
        outputs = cross_prod_cuda.forward(x1, x2)
        return outputs

class CrossSubtractFunction(Function):
    '''a-b'''
    @staticmethod
    def forward(ctx, x1, x2):
        outputs = cross_subtract_cuda.forward(x1, x2)
        return outputs

def cross_prod(pcl_1, pcl_2):
    prod = CrossProdFunction.apply(pcl_1, pcl_2)
    return prod

def cross_subtract(pcl_1, pcl_2):
    sub = CrossSubtractFunction.apply(pcl_1, pcl_2)
    return sub

'''
from monodepth2/cvo_utils.py
'''
# import cvo_dense_samp, cvo_dense_angle, cvo_dense_normal, cvo_dense_with_normal
# import cvo_dense_with_normal_output

class PtSampleInGridSigma(Function):
    """This is for calculating c3d loss for 2D points with covariance matrix for each flat point. 
    """
    @staticmethod
    def forward(ctx, pts, pts_info, pts_ells, grid_source, grid_valid, neighbor_range, ignore_ib=False, return_pdf=False):
        """ pts: 1*3*N (3 is u,v,b), pts_info: 1*2*N, pts_ells: 1*4*N (4 is sigma_x, sigma_y, rho_xy, weight), grid_source: B*2*H*W (2 is x and y), grid_valid: B*1*H*W, neighbor_range: int
        """
        outputs = cvo_dense_Sigma.forward(pts, pts_info, pts_ells, grid_source, grid_valid, neighbor_range, ignore_ib, return_pdf)
        ctx.save_for_backward(pts, pts_info, pts_ells, grid_source, grid_valid)
        ctx.neighbor_range = neighbor_range
        ctx.ignore_ib = ignore_ib
        ctx.return_pdf = return_pdf
        return outputs

    @staticmethod
    def backward(ctx, dy):
        # outputs, pts, pts_info, grid_source, grid_valid = ctx.saved_tensors
        pts, pts_info, pts_ells, grid_source, grid_valid = ctx.saved_tensors
        dy = dy.contiguous()
        dx1, dx2, dells = cvo_dense_Sigma.backward(dy, pts, pts_info, pts_ells, grid_source, grid_valid, ctx.neighbor_range, ctx.ignore_ib, ctx.return_pdf)
        return None, dx1, dells, dx2, None, None, None, None

class PtSampleInGridSigmaGrid(Function):
    """This is for calculating c3d loss for 2D points with covariance matrix for each grid point. 
    """
    @staticmethod
    def forward(ctx, pts, pts_info, grid_ells, grid_source, grid_valid, neighbor_range, ignore_ib=False, return_pdf=False):
        """ pts: 1*3*N (3 is u,v,b), pts_info: 1*2*N, grid_ells: B*4*N (4 is sigma_x, sigma_y, rho_xy, weight), grid_source: B*2*H*W (2 is x and y), grid_valid: B*1*H*W, neighbor_range: int
        """
        outputs = cvo_dense_Sigma_grid.forward(pts, pts_info, grid_ells, grid_source, grid_valid, neighbor_range, ignore_ib, return_pdf)
        ctx.save_for_backward(pts, pts_info, grid_ells, grid_source, grid_valid)
        ctx.neighbor_range = neighbor_range
        ctx.ignore_ib = ignore_ib
        ctx.return_pdf = return_pdf
        return outputs

    @staticmethod
    def backward(ctx, dy):
        # outputs, pts, pts_info, grid_source, grid_valid = ctx.saved_tensors
        pts, pts_info, grid_ells, grid_source, grid_valid = ctx.saved_tensors
        dy = dy.contiguous()
        dx1, dx2, dells = cvo_dense_Sigma_grid.backward(dy, pts, pts_info, grid_ells, grid_source, grid_valid, ctx.neighbor_range, ctx.ignore_ib, ctx.return_pdf)
        return None, dx1, dells, dx2, None, None, None, None

# class PtSampleInGridSigmaMuGrid(Function):
#     @staticmethod
#     def forward(ctx, pts, pts_info, grid_ells, grid_source, grid_valid, neighbor_range, ignore_ib=False):
#         """ pts: B*2*N, pts_info: B*C*N, grid_source: B*C*H*W (C could be xyz, rgb, ...), grid_valid: B*1*H*W, neighbor_range: int
#         """
#         outputs = cvo_dense_Sigma_mu_grid.forward(pts, pts_info, grid_ells, grid_source, grid_valid, neighbor_range, ignore_ib)
#         ctx.save_for_backward(pts, pts_info, grid_ells, grid_source, grid_valid)
#         ctx.neighbor_range = neighbor_range
#         ctx.ignore_ib = ignore_ib
#         return outputs

#     @staticmethod
#     def backward(ctx, dy):
#         # outputs, pts, pts_info, grid_source, grid_valid = ctx.saved_tensors
#         pts, pts_info, grid_ells, grid_source, grid_valid = ctx.saved_tensors
#         dy = dy.contiguous()
#         dx1, dx2, dells = cvo_dense_Sigma_mu_grid.backward(dy, pts, pts_info, grid_ells, grid_source, grid_valid, ctx.neighbor_range, ctx.ignore_ib)
#         return None, dx1, dells, dx2, None, None, None
        
class PtSampleInGrid(Function):
    @staticmethod
    def forward(ctx, pts, pts_info, grid_source, grid_valid, neighbor_range, ell, ignore_ib=False, sqr=True, ell_basedist=0, return_pdf=False):
        """ pts: 1*3*N (3 is u,v,b), pts_info: 1*C*N, grid_source: B*C*H*W (C could be xyz, rgb, ...), grid_valid: B*1*H*W, neighbor_range: int
        return_pdf: if True, return pdf except sqrt((2pi)^k) constant, otherwise only the exp part (exponential kernel). Default false. Only valid if sqr is True. 
        """
        assert not (return_pdf and not sqr), "return_pdf is valid only when sqr is True"
        outputs = cvo_dense_samp.forward(pts, pts_info, grid_source, grid_valid, neighbor_range, ell, ignore_ib, sqr, ell_basedist, return_pdf)
        # ctx.save_for_backward(outputs, pts, pts_info, grid_source, grid_valid)
        ctx.save_for_backward(pts, pts_info, grid_source, grid_valid)
        ctx.neighbor_range = neighbor_range
        ctx.ell = ell
        ctx.ignore_ib = ignore_ib
        ctx.sqr = sqr
        ctx.ell_basedist = ell_basedist
        ctx.return_pdf = return_pdf
        return outputs
    # def forward(ctx, pts, pts_info, grid_source, grid_valid, neighbor_range, ell, ignore_ib=False, sqr=True, ell_basedist=0):
    #     """ pts: B*2*N, pts_info: B*C*N, grid_source: B*C*H*W (C could be xyz, rgb, ...), grid_valid: B*1*H*W, neighbor_range: int
    #     dummy version for memory leak debug
    #     """
    #     outputs = torch.ones( (1,(2*neighbor_range+1)*(2*neighbor_range+1), pts.shape[-1]), device=pts_info.device )
    #     # ctx.save_for_backward(outputs, pts, pts_info, grid_source, grid_valid)
    #     ctx.save_for_backward(pts, pts_info, grid_source, grid_valid)
    #     ctx.neighbor_range = neighbor_range
    #     ctx.ell = ell
    #     ctx.ignore_ib = ignore_ib
    #     ctx.sqr = sqr
    #     ctx.ell_basedist = ell_basedist
    #     return outputs

    @staticmethod
    def backward(ctx, dy):
        # outputs, pts, pts_info, grid_source, grid_valid = ctx.saved_tensors
        pts, pts_info, grid_source, grid_valid = ctx.saved_tensors
        dy = dy.contiguous()
        # dx1, dx2 = cvo_dense_samp.backward(dy, outputs, pts, pts_info, grid_source, grid_valid, ctx.neighbor_range, ctx.ell, ctx.ignore_ib)
        dx1, dx2 = cvo_dense_samp.backward(dy, pts, pts_info, grid_source, grid_valid, ctx.neighbor_range, ctx.ell, ctx.ignore_ib, ctx.sqr, ctx.ell_basedist, ctx.return_pdf)
        return None, dx1, dx2, None, None, None, None, None, None, None
    # def backward(ctx, dy): ## dummy version for memory leak debug
    #     # outputs, pts, pts_info, grid_source, grid_valid = ctx.saved_tensors
    #     pts, pts_info, grid_source, grid_valid = ctx.saved_tensors
    #     dy = dy.contiguous()
    #     # dx1, dx2 = cvo_dense_samp.backward(dy, outputs, pts, pts_info, grid_source, grid_valid, ctx.neighbor_range, ctx.ell, ctx.ignore_ib)
    #     # dx1, dx2 = cvo_dense_samp.backward(dy, pts, pts_info, grid_source, grid_valid, ctx.neighbor_range, ctx.ell, ctx.ignore_ib, ctx.sqr, ctx.ell_basedist)
    #     dx1 = torch.zeros_like(pts_info)
    #     dx2 = torch.zeros_like(grid_source)
    #     return None, dx1, dx2, None, None, None, None, None, None


class PtSampleInGridAngle(Function):
    @staticmethod
    def forward(ctx, pts, pts_info, grid_source, grid_valid, neighbor_range, ignore_ib=False):
        """ pts: B*2*N, pts_info: B*C*N, grid_source: B*C*H*W (C could be xyz, rgb, ...), grid_valid: B*1*H*W, neighbor_range: int
        """
        outputs = cvo_dense_angle.forward(pts, pts_info, grid_source, grid_valid, neighbor_range, ignore_ib)
        ctx.save_for_backward(pts, pts_info, grid_source, grid_valid)
        ctx.neighbor_range = neighbor_range
        ctx.ignore_ib = ignore_ib
        return outputs

    @staticmethod
    def backward(ctx, dy):
        pts, pts_info, grid_source, grid_valid = ctx.saved_tensors
        dy = dy.contiguous()
        dx1, dx2 = cvo_dense_angle.backward(dy, pts, pts_info, grid_source, grid_valid, ctx.neighbor_range, ctx.ignore_ib)
        return None, dx1, dx2, None, None, None, None


class PtSampleInGridWithNormal(Function):
    @staticmethod
    def forward(ctx, pts, pts_info, grid_source, grid_valid, pts_normal, grid_normal, pts_nres, grid_nres, neighbor_range, ell, mag_max, mag_min, ignore_ib=False, norm_in_dist=False, neg_nkern_to_zero=False, ell_basedist=0, 
                return_nkern=False, filename=""):
        """ pts: B*2*N, pts_info: B*C*N, grid_source: B*C*H*W (C could be xyz, rgb, ...), grid_valid: B*1*H*W, neighbor_range: int
        """
        if not return_nkern:
            # y = cvo_dense_with_normal.forward(pts, pts_info, grid_source, grid_valid, pts_normal, grid_normal, pts_nres, grid_nres, neighbor_range, ell, mag_max, mag_min, ignore_ib, norm_in_dist, ell_basedist)
            outputs = cvo_dense_with_normal_output.forward(pts, pts_info, grid_source, grid_valid, pts_normal, grid_normal, pts_nres, grid_nres, neighbor_range, ell, mag_max, mag_min, ignore_ib, norm_in_dist, neg_nkern_to_zero, ell_basedist, return_nkern)
            y = outputs[0]
        else:
            outputs = cvo_dense_with_normal_output.forward(pts, pts_info, grid_source, grid_valid, pts_normal, grid_normal, pts_nres, grid_nres, neighbor_range, ell, mag_max, mag_min, ignore_ib, norm_in_dist, neg_nkern_to_zero, ell_basedist, return_nkern)
            y = outputs[0]
            nkern = outputs[1]
            save_nkern(nkern, pts, grid_source.shape, mag_max, mag_min, filename)

        ctx.save_for_backward(pts, pts_info, grid_source, grid_valid, pts_normal, grid_normal, pts_nres, grid_nres)
        ctx.neighbor_range = neighbor_range
        ctx.ell = ell
        ctx.mag_max = mag_max
        ctx.mag_min = mag_min
        ctx.ignore_ib = ignore_ib
        ctx.norm_in_dist = norm_in_dist
        ctx.neg_nkern_to_zero = neg_nkern_to_zero
        ctx.ell_basedist =ell_basedist
        return y

    @staticmethod
    def backward(ctx, dy):
        pts, pts_info, grid_source, grid_valid, pts_normal, grid_normal, pts_nres, grid_nres = ctx.saved_tensors
        dy = dy.contiguous()
        # dx1, dx2, dn1, dn2, dr1, dr2 = cvo_dense_with_normal.backward( \
        #     dy, pts, pts_info, grid_source, grid_valid, pts_normal, grid_normal, pts_nres, grid_nres, ctx.neighbor_range, ctx.ell, ctx.mag_max, ctx.mag_min, ctx.ignore_ib, ctx.norm_in_dist, ctx.ell_basedist)
        dx1, dx2, dn1, dn2, dr1, dr2 = cvo_dense_with_normal_output.backward( \
            dy, pts, pts_info, grid_source, grid_valid, pts_normal, grid_normal, pts_nres, grid_nres, ctx.neighbor_range, ctx.ell, ctx.mag_max, ctx.mag_min, ctx.ignore_ib, ctx.norm_in_dist, ctx.neg_nkern_to_zero, ctx.ell_basedist)
        # return None, dx1, dx2, None, dn1, dn2, dr1, dr2, None, None, None, None, None, None, None, None, None, None
        # return None, dx1, dx2, None, None, None, dr1, dr2, None, None, None, None, None, None, None, None, None, None
        return None, dx1, dx2, None, dn1, dn2, None, None, None, None, None, None, None, None, None, None, None, None

class PtSampleInGridCalcNormal(Function):
    @staticmethod
    def forward(ctx, pts, grid_source, grid_valid, neighbor_range, ignore_ib):
        normals, norm_sq, ioffs = cvo_dense_normal.forward(pts, grid_source, grid_valid, neighbor_range, ignore_ib)
        ctx.save_for_backward(ioffs, pts, grid_source)
        ctx.ignore_ib = ignore_ib
        return normals, norm_sq
    
    @staticmethod
    def backward(ctx, dnormals, dnorms):
        ioffs, pts, grid_source = ctx.saved_tensors
        dgrid = cvo_dense_normal.backward(dnormals, dnorms, ioffs, pts, grid_source, ctx.ignore_ib)
        return None, dgrid, None, None, None
        # return None, None, None, None, None
