### originally in monodepth2
try:
    import pcl
except:
    import warnings
    warnings.warn("Optional: install pcl library at https://github.com/cmpute/pcl.py to enable point cloud visualization.")
import numpy as np
import torch

from .color import rgbmap

# cloud = pcl.PointCloud(np.array([[1,2,3,0]], dtype='f4')) # float32, n*4
# cloud = pcl.create_xyz(array) # n*3

# cloud = pcl.create_xyzrgb(array) # n*6
# cloud.to_ndarray()

# vis = pcl.Visualizer()
# vis.addPointCloud(cloud)
# vis.addCoordinateSystem()

# vis.spin()

############################### pcl construction ##############################

def pcl_from_np_single(xyz, rgb=None, intensity=None, normal=None):
    """accept np.ndarray N*C
    intensity can have channel dimension or not."""
    ### to desired shape and scale
    assert xyz.shape[1] == 3
    if rgb is not None:
        assert rgb.shape[1] == 3
        if rgb.max() <= 1:
            rgb = rgb * 255
    if intensity is not None:
        if intensity.ndim == 1:
            intensity = intensity.reshape(-1,1)
        assert intensity.shape[1] == 3
        if intensity.max() <= 1:
            intensity = intensity * 255
    if normal is not None:
        assert normal.shape[1] == 3

    ### construct pcl objects
    if rgb is not None:
        xyz_rgb = np.concatenate((xyz, rgb), axis=1)
        cloud = pcl.create_xyzrgb(xyz_rgb)
    elif intensity is not None:
        xyz_inten = np.concatenate((xyz, intensity), axis=1)
        cloud = pcl.create_xyzi(xyz_inten)
    else:
        cloud = pcl.create_xyz(xyz)

    if normal is not None:
        cloud_nm = pcl.create_normal(normal)
        cloud = cloud.append_fields(cloud_nm)

    return cloud

def pcl_from_flat_xyz(xyz, rgb=None, intensity=None, normal=None):
    """accept torch.Tensor of (B*)C*N or np.ndarray (B*)N*C
    intensity can have channel dimension or not.
    if batched, return a list of cloud"""
    if xyz.ndim == 3:
        batched = True
    else:
        batched = False
    
    if not isinstance(xyz, np.ndarray):
        xyz = xyz.detach().cpu().numpy().swapaxes(-1, -2)
        if rgb is not None:
            rgb = rgb.detach().cpu().numpy().swapaxes(-1, -2)
        if intensity is not None:
            intensity = intensity.detach().cpu().numpy()
            inten_has_c = intensity.ndim==3 if batched else intensity.ndim==2
            if inten_has_c:
                intensity = intensity.swapaxes(-1, -2)
        if normal is not None:
            normal = normal.detach().cpu().numpy().swapaxes(-1, -2)
    
    if batched:
        clouds = []
        for ib in range(xyz.shape[0]):
            xyz_i = xyz[ib]
            rgb_i = rgb[ib] if rgb is not None else None
            intensity_i = intensity[ib] if intensity is not None else None
            normal_i = normal[ib] if normal is not None else None
            cloud_i = pcl_from_np_single(xyz_i, rgb_i, intensity_i, normal_i)
            clouds.append(cloud_i)
        return clouds
    else:
        return pcl_from_np_single(xyz, rgb, intensity, normal)

def pcl_from_grid_xy1_dep(xy1, depth, rgb=None, intensity=None, normal=None):
    """accept torch.Tensor of (B*)C*H*W or np.ndarray (B*)H*W*C
    depth and intensity can have channel dimension or not.
    mask comes from depth (valid with positive value)
    if batched, return a list of cloud"""

    if xy1.ndim == 4:
        batched = True
    else:
        batched = False

    ### transform (B*)C*H*W tensor to (B*)H*W*C numpy array (for depth, transform to (B*)H*W )
    if not isinstance(xy1, np.ndarray):
        xy1 = np.moveaxis(xy1.cpu().numpy(), -3, -1)
        depth = depth.detach().cpu().numpy()
        depth_has_c = depth.ndim==4 if batched else depth.ndim==3
        if depth_has_c:
            depth = np.moveaxis(depth, -3, -1)[..., 0] ### remove the channel dim for depth so that mask does not have channel dim
        if rgb is not None:
            rgb = np.moveaxis(rgb.detach().cpu().numpy(), -3, -1)
        if intensity is not None:
            intensity = intensity.detach().cpu().numpy()
            inten_has_c = intensity.ndim==4 if batched else intensity.ndim==3
            if inten_has_c:
                intensity = np.moveaxis(intensity, -3, -1)
        if normal is not None:
            normal = np.moveaxis(normal.detach().cpu().numpy(), -3, -1)
    
    if batched:
        clouds = []
        for ib in range(xy1.shape[0]):
            mask_i = depth[ib] > 0 # H*W
            depth_i = depth[ib][mask_i]
            xy1_i = xy1[ib][mask_i]
            xyz_i = xy1_i * depth_i[...,None]
            rgb_i = rgb[ib][mask_i] if rgb is not None else None
            intensity_i = intensity[ib][mask_i] if intensity is not None else None
            normal_i = normal[ib][mask_i] if normal is not None else None
            cloud_i = pcl_from_np_single(xyz_i, rgb_i, intensity_i, normal_i)
            clouds.append(cloud_i)
        return clouds
    else:
        mask_i = depth > 0 # H*W
        depth_i = depth[mask_i]
        xy1_i = xy1[mask_i]
        xyz_i = xy1_i * depth_i[...,None]
        rgb_i = rgb[mask_i] if rgb is not None else None
        intensity_i = intensity[mask_i] if intensity is not None else None
        normal_i = normal[mask_i] if normal is not None else None
        return pcl_from_np_single(xyz_i, rgb_i, intensity_i, normal_i)

############################### pcl io ##############################

def pcl_write(cloud, filename):
    pcl.io.save_pcd('{}.pcd'.format(filename), cloud)

def pcl_read(filename):
    cloud = pcl.io.load_pcd(filename)
    return cloud

############################### pcl editing ##############################

def pcl_clip_distance(cloud, max_dist):
    """only for XYZRGB pcl object"""
    array = cloud.to_ndarray()
    valid_idx = array['z'] < max_dist
    array_sub = array[valid_idx]

    cloud_dt = np.dtype(dict(names=['x','y','z','rgb'], formats=['f4','f4','f4','u4'], offsets=[0,4,8,16], itemsize=20))
    cloud_arr = np.empty(len(array_sub), dtype=cloud_dt)
    cloud_arr['x'], cloud_arr['y'], cloud_arr['z'], cloud_arr['rgb'] = array_sub['x'], array_sub['y'], array_sub['z'], array_sub['rgb']
    cloud_sub = pcl.PointCloud(cloud_arr, 'XYZRGB')

    return cloud_sub

def pcl_xyzi2xyzrgb(cloud):
    array = cloud.to_ndarray()
    
    x = np.array([arr[0] for arr in array])
    y = np.array([arr[1] for arr in array])
    z = np.array([arr[2] for arr in array])

    inten = np.array([arr[3] for arr in array])
    # print(inten.max())
    # print(inten.min())
    r,g,b = rgbmap(inten)

    xyzrgb = np.stack([x,y,z,r, g, b], axis=-1)
    # print(xyzrgb.shape)
    cloudrgb = pcl.create_xyzrgb(xyzrgb)
    return cloudrgb

############################### pcl visualization ##############################

def pcl_load_viewpoint(filename):
    """accept a .cam file generated by pcl, return pos, view, focal, which are arguments to setCameraPosition() of pcl.Visualizer
    win_size is argument to setSize() of pcl.Visualizer """
    with open(filename) as f:
        line = f.readline()
    blocks = line.split("/")
    clip, focal, pos, view, angle, win_size, win_pos = [[float(x) for x in block.split(",")] for block in blocks]

    #   ofs_cam << clip[0]  << "," << clip[1]  << "/" << focal[0] << "," << focal[1] << "," << focal[2] << "/" <<
    #          pos[0]   << "," << pos[1]   << "," << pos[2]   << "/" << view[0]  << "," << view[1]  << "," << view[2] << "/" <<
    #          cam->GetViewAngle () / 180.0 * M_PI  << "/" << win_size[0] << "," << win_size[1] << "/" << win_pos[0] << "," << win_pos[1]
    #       << std::endl;
    # https://github.com/PointCloudLibrary/pcl/blob/000a762bb84781a119ee0cd4ba6bdfee59325fc3/visualization/src/interactor_style.cpp#L136

    return pos, view, focal, win_size

def pcl_load_viewer_fromfile(filename=None):
    """accept a .cam file generated by pcl, generate a pcl.Visualizer"""

    viewer = pcl.Visualizer()
    viewer.addCoordinateSystem()   # optional

    if filename is not None:
        pos, view, focal, win_size = pcl_load_viewpoint(filename)
        viewer.setCameraPosition(pos, view, focal) #https://github.com/PointCloudLibrary/pcl/blob/000a762bb84781a119ee0cd4ba6bdfee59325fc3/visualization/src/interactor_style.cpp 
        # viewer.setCameraClipDistances(0.0001,5.0)
        viewer.setSize(win_size[0], win_size[1])

    return viewer

def pcl_vis_seq(clouds, viewer=None, cam_fname=None, snapshot_fname_fmt=None, vis_normal=False, block=False):
    """if viewer is not given, initialize one from cam_fname file.
    If cam_fname is None, initialize a default viewer.
    If snapshot_fname_fmt is not None, write snapshot of the point cloud to png file.
    snapshot_fname_fmt can be a list, with names already generated for each of cloud in the list. 
    snapshot_fname_fmt can be a string with placeholder "{}", filled with index of the cloud in the list. 
    """
    new_viewer_flag = False
    if viewer is None:
        viewer = pcl_load_viewer_fromfile(cam_fname)
        new_viewer_flag = True

    pt_size = 3
    
    for i, cloud in enumerate(clouds):
        viewer.removeAllPointClouds()
        viewer.addPointCloud(cloud, "cloud")  # give a name if going to add more than one point cloud in the visualizer, or need to change proterties of this cloud
        viewer.setPointCloudRenderingProperties("PointSize", pt_size, id="cloud")   # optional, to set the point cloud size
        if vis_normal:
            viewer.addPointCloudNormals(cloud, level=10, scale=0.1, id='normal') # 'pred_grad_dir'

        if snapshot_fname_fmt is not None:
            if isinstance(snapshot_fname_fmt, list):
                snap_fname = snapshot_fname_fmt[i]
            else:
                snap_fname = snapshot_fname_fmt.format(i)
            viewer.saveScreenshot(snap_fname+".png")
            
        if not block:
            viewer.spinOnce()       # do not stop
        else:
            viewer.spin()         # stuck every frame

    if new_viewer_flag:
        viewer.close()

def pcl_vis(clouds):
    if isinstance(clouds, list):
        for cloud in clouds:
            vis = pcl.Visualizer()
            vis.addPointCloud(cloud)
            # if normal is not None:
            #     vis.addPointCloudNormals(cloud, cloud_nm)
            # else:
            vis.addCoordinateSystem()
            vis.spin()
    else:
        vis = pcl.Visualizer()
        vis.addPointCloud(clouds)
        vis.addCoordinateSystem()
        vis.spin()