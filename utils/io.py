import torch
import numpy as np 
from PIL import Image

from .pc3d import grid_from_concat_flat_func

'''
from monodepth2/cvo_utils.py
'''
def save_nkern(nkern, pts, grid_shape, mag_max, mag_min, filename):
    """nkern is a 1*NN*N tensor, need to turn it into grid and save as image"""
    pts_coords = pts.to(dtype=torch.long).squeeze(0).transpose(0,1).split(1,dim=1)
    list_spn = []

    dim_n = int(np.sqrt(nkern.shape[1]))
    half_dim_n = int( (dim_n-1)/2 )
    list_spn.append( half_dim_n )

    mid_n = ( nkern.shape[1]-1 ) / 2
    mid_n = int(mid_n)
    list_spn.append(mid_n)
    list_spn.append(mid_n - half_dim_n)
    list_spn.append(mid_n + half_dim_n)
    list_spn.append( int(nkern.shape[1] - 1 - half_dim_n) )

    for spn in list_spn:
        nkern_center = nkern[:, spn:spn+1, :] ## only take one slice 1*1*N
        nkern_center_grid = grid_from_concat_flat_func(pts_coords, nkern_center, grid_shape) # B*1*H*W

        nkern_center_grid = nkern_center_grid.squeeze(1).cpu().numpy() # B*H*W
        nkern_center_grid = (nkern_center_grid / mag_max * 255).astype(np.uint8) # normalize to 0-255

        for ib in range(nkern_center_grid.shape[0]):
            img = Image.fromarray(nkern_center_grid[ib])
            img.save( "{}_{}_{}.png".format(filename, spn, ib) )
            
'''
from monodepth2/cvo_utils.py
'''
def save_tensor_to_img(tsor, filename, mode):
    """Input is B*C*H*W"""
    nparray = tsor.cpu().detach().numpy()
    nparray = nparray.transpose(0,2,3,1)
    if "rgb" in mode:
        nparray = (nparray * 255).astype(np.uint8)
        Imode = "RGB"
    elif "dep" in mode:
        nparray = (nparray[:,:,:,0] /nparray.max() * 255).astype(np.uint8)
        # nparray = (nparray[:,:,:,0] * 255).astype(np.uint8) # disable normalization since disp is already in [0, 1]
        Imode = "L"
    elif "nml" in mode:
        nparray = (nparray * 255).astype(np.uint8)
        Imode = "RGB"
    else:
        raise ValueError("mode {} not recognized".format(mode))
    for ib in range(nparray.shape[0]):
        img = Image.fromarray(nparray[ib], mode=Imode)
        img.save("{}_{}_{}.png".format(filename, mode, ib))