import torch
import torch.nn.functional as F

from ..cvo_funcs import *

def get_stat_lidar_dist(norm_sq):   # 4*2*N
    norm_L2 = torch.sqrt(norm_sq)
    norm_L2 = norm_L2.reshape(8, -1) # 8*N
    norm_L2_max, _ = norm_L2.max(dim=0, keepdim=True)

    norm_L2_altered = norm_L2.clone()
    norm_L2_altered[norm_L2_altered==0] = 1000
    norm_L2_altered[:, norm_L2_max[0]==0] = 0
    norm_L2_min, _ = norm_L2_altered.min(dim=0, keepdim=True)

    norm_L2_mean = norm_L2.sum(dim=0, keepdim=True) / ((norm_L2 > 0).sum(dim=0, keepdim=True).to(dtype=torch.float32)+1e-8) # 1*N

    norm_L2_stat = torch.stack([norm_L2_min, norm_L2_max, norm_L2_mean], dim=1) # 1*3*N
    return norm_L2_stat

'''from monodepth2/cvo_utils.py'''
def recall_grad(pre_info, grad):
    # print(pre_info, grad)
    # print(pre_info, torch.isnan(grad).any())
    assert not torch.isnan(grad).any(), pre_info

'''from monodepth2/cvo_utils.py'''
def calc_normal(pts, grid_source, grid_valid, neighbor_range, ignore_ib, min_dist_2=0.05, return_stat=False):
    raw_normals, norm_sq = PtSampleInGridCalcNormal.apply(pts.contiguous(), grid_source.contiguous(), grid_valid.contiguous(), neighbor_range, ignore_ib) ## raw_normals is 4*C*N, and norm_sq is 4*2*N

    if raw_normals.requires_grad:
        raw_normals.register_hook(lambda grad: recall_grad("raw_normals", grad) )

    # raw_normals = torch.ones((4,3,pts.shape[-1]), device=grid_source.device, dtype=grid_source.dtype)
    # norm_sq = torch.ones((4,2,pts.shape[-1]), device=grid_source.device, dtype=grid_source.dtype)

    ##################### inspection of lidar point density
    if return_stat:
        norm_L2_stat = get_stat_lidar_dist(norm_sq)
    #######################################################

    normed_normal = F.normalize(raw_normals, p=2, dim=1) # 4*C*N

    norms = torch.sqrt(norm_sq + 1e-8) # |a|, |b|, 4*2*N
    normal_sin_scale = raw_normals / (norms[:,0:1] * norms[:,1:2]) # raw_normal |axb| = |a||b|sin(alpha), 4*C*N

    W_norms = 1 / ( torch.clamp(norms, min=min_dist_2).sum(dim=1, keepdim=True) ) # 4*1*N
    weighted_normal = (normal_sin_scale * W_norms).sum(dim=0, keepdim=True) # 1*C*N
    weighted_normal = F.normalize(weighted_normal, p=2, dim=1) # weighted_normal / torch.norm(weighted_normal, dim=0, keepdim=True) # 1*C*N

    W_norms_effective = torch.norm(normal_sin_scale, dim=1,keepdim=True) * W_norms # 4*1*N
    W_norms_sum = W_norms_effective.sum(dim=0, keepdim=True) # 1*1*N
    
    ## calculate residual
    res_sin_sq = 1- (normed_normal * weighted_normal).sum(dim=1, keepdim=True).pow(2) # 4*1*N

    ## the F.normalize will generally result in norm slightly larger than 1
    res_sin_sq = res_sin_sq.clamp(min=0)

    # with torch.no_grad():
    #     diff_normal = (normed_normal - weighted_normal).norm(dim=1) # 4*N
    #     weit_normal = weighted_normal.norm(dim=1) # 1*N
    #     print("identical #:", float(((diff_normal==0) & (weit_normal!=0)).sum() ) )

    #     single = ((raw_normals.norm(dim=1) > 0).sum(dim=0))==1 # N
    #     select_normal = diff_normal[:,single] # 4*N_sub
    #     selsel_normal = select_normal.min(dim=0)[0] # N_sub
    #     print(float(selsel_normal.min()), float(selsel_normal.max()))
        
    #     print("single #:", float(single.sum()))
    #     print("..............................")

    #     normed_norm = normed_normal.norm(dim=1)
    #     normed_normw = weighted_normal.norm(dim=1)

    #     print("normed_norm", float(normed_norm.min()), float(normed_norm.max()))
    #     print("normed_normw", float(normed_normw.min()), float(normed_normw.max()))
    #     print("res_sin_sq", float(res_sin_sq.min()), float(res_sin_sq.max()))

    res_weighted_sum = (res_sin_sq * W_norms_effective).sum(dim=0, keepdim=True) / (W_norms_sum + 1e-8) # 1*1*N

    single = ((raw_normals.norm(dim=1) > 0).sum(dim=0))==1 # N # the points whose normal is calculated from cross product of only 1 pair of points
    single_cross = single.view(1,1,-1)
    # single_cross = (W_norms_sum != 0) & (res_weighted_sum == 0) # 1*1*N
    single_cross_default_sin_sq = torch.ones_like(res_weighted_sum) * 0.5 # 1*1*N
    res_final = torch.where(single_cross, single_cross_default_sin_sq, res_weighted_sum) # 1*1*N

    ## weighted_normal is unit length normal vectors 1*C*N; res_final in [0,1], 1*1*N
    ## some vectors in weighted_normal could be zero if no neighboring pixels are found
    if return_stat:
        return weighted_normal, res_final, norm_L2_stat
    else:
        return weighted_normal, res_final

'''from monodepth2/cvo_utils.py'''
def res_normal_dense(xyz, normal, K):
    """
    xyz: B*C*H*W, C=3
    normal: B*C*H*W (normalized), C=3
    K: cam intr
    """
    batch_size = xyz.shape[0]
    channel = xyz.shape[1]
    xyz_patches = F.unfold(xyz, kernel_size=3, padding=1).reshape(batch_size, channel, 9, -1)    # B*(C*9)*(H*W) -> B*C*9*(H*W)
    xyz_patch_proj = (xyz_patches * normal.reshape(batch_size, channel, 1, -1 )).sum(dim=1)  # B*9*(H*W)
    xyz_patch_proj_res = xyz_patch_proj[:, [0,1,2,3,5,6,7,8], :] - xyz_patch_proj[:,[4], :] # B*8*(H*W)
    xyz_patch_diff = ( xyz_patches[:, :, [0,1,2,3,5,6,7,8], :] - xyz_patches[:, :, [4], :] ).norm(dim=1)  # B*8*(H*W)
    
    xyz_patch_res_sin = ( xyz_patch_proj_res/ (xyz_patch_diff+1e-8) ).abs().mean(dim=1, keepdim=True).reshape(batch_size, 1, xyz.shape[2], xyz.shape[3]) # B*1*H*W between 0~1

    return xyz_patch_res_sin

'''from monodepth2/cvo_utils.py'''
class NormalFromDepthDense(torch.nn.Module):
    def __init__(self):
        super(NormalFromDepthDense, self).__init__()
        self.sobel_grad = SobelGrad()

    def forward(self, depth, K):
        grad_x, grad_y = self.sobel_grad(depth)
        normal = normal_from_grad(grad_x, grad_y, depth, K)
        # tan_x = tan_from_grad(grad_x, depth, K, mode="x")
        # tan_y = tan_from_grad(grad_x, depth, K, mode="y")
        # normal = normal_from_tan(tan_x, tan_y)
        return normal


'''from monodepth2/cvo_utils.py'''
class SobelGrad(torch.nn.Module):
    def __init__(self):
        super(SobelGrad, self).__init__()
        filter_shape = (1, 1, 3, 3) # out_c, in_c/group, kH, kW
        kern_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(filter_shape) / 8.0 # normalize so that the value is a real gradient delta(d)/delta(x)
        kern_y = kern_x.transpose(2,3)
        self.register_buffer("kern_x", kern_x)
        self.register_buffer("kern_y", kern_y)
        # self.pad_layer = torch.nn.ReflectionPad2d(1) # (left,right,top,bottom) or an int
        self.pad_layer = torch.nn.ReplicationPad2d(1)
        ## use a dedicated padding layer because the padding in F.conv2d only pads zeros.

    def forward(self, img):
        """expect the img channel to be 1 if used in NormalFromDepthDense, 
        Otherwise the return channel number is the same as input
        """
        img_pad = self.pad_layer(img)
        img_pad[:,:,-1, :] = 2 * img_pad[:,:,-2, :] - img_pad[:,:,-3, :]        ## so that the last row's vertical gradient is decided by the last two rows
        grad_x = torch.zeros_like(img)
        grad_y = torch.zeros_like(img)
        for ic in range(img.shape[1]):
            grad_x[:,ic:ic+1,:,:] = F.conv2d(img_pad[:,ic:ic+1,:,:], self.kern_x)
            grad_y[:,ic:ic+1,:,:] = F.conv2d(img_pad[:,ic:ic+1,:,:], self.kern_y)
        
        return grad_x, grad_y
     
'''from monodepth2/cvo_utils.py'''       
def tan_from_grad(grad, depth, K, mode):
    """
    grad: grad_x or grad_y, B*1*H*W
    depth: B*1*H*W
    K: cam intrinsic mat
    mode: "x" or "y"
    """
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]
    y_range = torch.arange(grad.shape[2], device=grad.device, dtype=grad.dtype) # height, y, v
    x_range = torch.arange(grad.shape[3], device=grad.device, dtype=grad.dtype) # width, x, u
    grid_y, grid_x = torch.meshgrid(y_range, x_range) ## [height * width]
    
    ## x_hat and y_hat
    x_hat = (grid_x - cx) / fx  # [h*w]
    y_hat = (grid_y - cy) / fy

    if mode== "x":
        tan_0 = x_hat * grad + depth / fx     #B*1*H*W
        tan_1 = y_hat * grad
        tan_2 = grad
        tan = torch.cat( (tan_0, tan_1, tan_2), dim=1 ) # B*3*H*W
    elif mode== "y":
        tan_0 = x_hat * grad    #B*1*H*W
        tan_1 = y_hat * grad + depth / fy
        tan_2 = grad
        tan = torch.cat( (tan_0, tan_1, tan_2), dim=1 ) # B*3*H*W
    else:
        raise ValueError("mode {} not recognized".format(mode))

'''from monodepth2/cvo_utils.py'''
def normal_from_tan(tan_x, tan_y):
    normal = torch.cross(tan_x, tan_y, dim=1) #B*3*H*W
    return F.normalize(normal, p=2, dim=1)

'''from monodepth2/cvo_utils.py'''
def normal_from_grad(grad_x, grad_y, depth, K):
    """grad_x: B*1*H*W"""
    y_range = torch.arange(grad_x.shape[2], device=grad_x.device, dtype=grad_x.dtype) # height, y, v
    x_range = torch.arange(grad_x.shape[3], device=grad_x.device, dtype=grad_x.dtype) # width, x, u
    grid_y, grid_x = torch.meshgrid(y_range, x_range) ## H*W

    fx = K[:,0,0].reshape(-1, 1, 1, 1) # B*1*1*1
    fy = K[:,1,1].reshape(-1, 1, 1, 1)
    cx = K[:,0,2].reshape(-1, 1, 1, 1)
    cy = K[:,1,2].reshape(-1, 1, 1, 1)

    normal_0 = -fx * grad_x
    normal_1 = -fy * grad_y
    normal_2 = (grid_x - cx) * grad_x + (grid_y - cy) * grad_y + depth
    normal = torch.cat([normal_0, normal_1, normal_2], dim=1)
    return F.normalize(normal, p=2, dim=1)