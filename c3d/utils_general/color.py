import torch
import numpy as np

'''from pytorch-unet/geometry.py'''
def rgb_to_hsv(image, flat=False):
    """Convert an RGB image to HSV.

    Args:
        input (torch.Tensor): RGB Image to be converted to HSV.
        flat: True if input B*C*N, False if input B*C*H*W

    Returns:
        torch.Tensor: HSV version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if not flat:
        if len(image.shape) < 3 or image.shape[-3] != 3:
            raise ValueError("Input size must have a shape of (*, 3, H, W) given flat=False. Got {}"
                            .format(image.shape))
    else:
        if len(image.shape) < 2 or image.shape[-2] != 3:
            raise ValueError("Input size must have a shape of (*, 3, N) given flat=True. Got {}"
                            .format(image.shape))

    if not flat:
        r: torch.Tensor = image[..., 0, :, :]
        g: torch.Tensor = image[..., 1, :, :]
        b: torch.Tensor = image[..., 2, :, :]

        maxc: torch.Tensor = image.max(-3)[0]
        minc: torch.Tensor = image.min(-3)[0]
    else:

        r: torch.Tensor = image[..., 0, :]
        g: torch.Tensor = image[..., 1, :]
        b: torch.Tensor = image[..., 2, :]

        maxc: torch.Tensor = image.max(-2)[0]
        minc: torch.Tensor = image.min(-2)[0]

    v: torch.Tensor = maxc  # brightness
    
    # ZMH: avoid division by zero
    v = torch.where(
        v == 0, torch.ones_like(v)*1e-3, v)

    deltac: torch.Tensor = maxc - minc
    s: torch.Tensor = deltac / v  # saturation

    # avoid division by zero
    deltac: torch.Tensor = torch.where(
        deltac == 0, torch.ones_like(deltac), deltac)

    rc: torch.Tensor = (maxc - r) / deltac
    gc: torch.Tensor = (maxc - g) / deltac
    bc: torch.Tensor = (maxc - b) / deltac

    maxg: torch.Tensor = g == maxc
    maxr: torch.Tensor = r == maxc

    h: torch.Tensor = 4.0 + gc - rc
    h[maxg]: torch.Tensor = 2.0 + rc[maxg] - bc[maxg]
    h[maxr]: torch.Tensor = bc[maxr] - gc[maxr]
    h[minc == maxc]: torch.Tensor = 0.0

    h: torch.Tensor = (h / 6.0) % 1.0

    if not flat:
        return torch.stack([h, s, v], dim=-3)
    else:
        return torch.stack([h, s, v], dim=-2)

'''from pytorch-unet/geometry.py'''
def hsv_to_rgb(image, flat=False):
    """Convert an HSV image to RGB.

    Args:
        input (torch.Tensor): HSV Image to be converted to RGB.
        flat: True if input B*C*N, False if input B*C*H*W

    Returns:
        torch.Tensor: RGB version of the image.
    https://gist.github.com/mathebox/e0805f72e7db3269ec22
    https://github.com/scikit-image/scikit-image/blob/master/skimage/color/colorconv.py#L214
    
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if not flat:
        if len(image.shape) < 3 or image.shape[-3] != 3:
            raise ValueError("Input size must have a shape of (*, 3, H, W) given flat=False. Got {}"
                            .format(image.shape))
    else:
        if len(image.shape) < 2 or image.shape[-2] != 3:
            raise ValueError("Input size must have a shape of (*, 3, N) given flat=True. Got {}"
                            .format(image.shape))

    if not flat:
        h: torch.Tensor = image[..., 0, :, :]
        s: torch.Tensor = image[..., 1, :, :]
        v: torch.Tensor = image[..., 2, :, :]
    else:
        h: torch.Tensor = image[..., 0, :]
        s: torch.Tensor = image[..., 1, :]
        v: torch.Tensor = image[..., 2, :]

    i = torch.floor(h*6)
    f = h*6 - i
    p = v * (1-s)
    q = v * (1-f*s)
    t = v * (1-(1-f)*s)

    rgbs = {}
    rgbs[0] = torch.stack((v, t, p), dim=1)
    rgbs[1] = torch.stack((q, v, p), dim=1)
    rgbs[2] = torch.stack((p, v, t), dim=1)
    rgbs[3] = torch.stack((p, q, v), dim=1)
    rgbs[4] = torch.stack((t, p, v), dim=1)
    rgbs[5] = torch.stack((v, p, q), dim=1)
    
    rgb = torch.zeros_like(image)
    iexpand = i.unsqueeze(1).expand_as(image)
    for idd in range(6):
        rgb = torch.where(iexpand == idd, rgbs[idd], rgb)
    rgb = torch.where(iexpand == 6, rgbs[0], rgb)

    # r, g, b = [
    #     (v, t, p),
    #     (q, v, p),
    #     (p, v, t),
    #     (p, q, v),
    #     (t, p, v),
    #     (v, p, q),
    # ][int(i%6)]

    return rgb

'''from bts/bts_utils.py'''
def rgbmap(gray, mask_zeros=False, max_val=-1, min_val=0):
    """
    Assuming input gray is 1D ndarray between [0,255]
    return is rgb between [0, 255]
    https://www.particleincell.com/2014/colormap/ 
    """
    # ##
    # r = inten
    # g = np.zeros_like(r)
    # b = 255- inten
    # ##

    if mask_zeros:
        valid_mask = gray>0
        invalid_mask = np.invert(valid_mask)
        min_cur = gray[valid_mask].min()
        max_cur = gray[valid_mask].max()
    else:
        min_cur = gray.min()
        max_cur = gray.max()
    
    if max_val == -1:
        ## mode 1: normalize to fulfill the range, excluding points of zero value (deal with them outside of this function)
        gray = (gray.astype(float) - min_cur )/( max_cur - min_cur )  # normalize to fulfill the range
    else:
        ## mode 2: normalize with fixed ratio
        gray = (gray.astype(float) - min_val )/( max_val - min_val )

    if mask_zeros:
        gray[invalid_mask] = 0.5 # this value does not matter as long as it will not cause out-of-range in calculating idx0. It will be set to 0 at the end

    gray_flat = gray.reshape(-1)

    a = (gray_flat)/0.25
    X = np.floor(a)           # group
    Y = np.floor( 255*(a-X) ) # residual
    cand = np.zeros((5,gray_flat.shape[0], 3))
    cand[0,:,0] = 255
    cand[0,:,1] = Y
    cand[0,:,2] = 0

    cand[1,:,0] = 255-Y
    cand[1,:,1] = 255
    cand[1,:,2] = 0

    cand[2,:,0] = 0
    cand[2,:,1] = 255
    cand[2,:,2] = Y

    cand[3,:,0] = 0
    cand[3,:,1] = 255-Y
    cand[3,:,2] = 255

    cand[4,:,0] = 0
    cand[4,:,1] = 0
    cand[4,:,2] = 255
    
    idx0 = X.astype(int)
    idx1 = np.arange(gray_flat.shape[0])

    rgb = cand[idx0, idx1]

    # gray = 0.3 + gray * 0.7
    # r = (rgb[:,0] * gray).astype(int)
    # g = (rgb[:,1] * gray).astype(int)
    # b = (rgb[:,2] * gray).astype(int)
    r = rgb[:,0].astype(int)
    g = rgb[:,1].astype(int)
    b = rgb[:,2].astype(int)

    r = r.reshape(gray.shape)
    g = g.reshape(gray.shape)
    b = b.reshape(gray.shape)

    if mask_zeros:
        r[invalid_mask] = 0
        g[invalid_mask] = 0
        b[invalid_mask] = 0
    
    return r,g,b