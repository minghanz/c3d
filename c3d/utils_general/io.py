import numpy as np 


'''from bts/c3d_loss.py'''
def read_calib_file(path):
    """Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data

def write_np_to_txt_like_kitti(f, array, name):
    if isinstance(array, (list, tuple, np.ndarray)):
        if isinstance(array, np.ndarray):
            array = array.reshape(-1)
        f.write("{}: ".format(name) + " ".join(str(x) for x in array) + "\n" )
    else:
        f.write("{}: ".format(name) + str(array) + "\n" )

'''from bts/bts_pre_intr.py'''
def load_velodyne_points(filename):
    """Load 3D point cloud from KITTI file format
    (adapted from https://github.com/hunse/kitti)
    """
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points