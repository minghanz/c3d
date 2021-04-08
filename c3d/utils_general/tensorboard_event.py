"""
This file contains util functions to work with event files used by tensorboard. 
"""

import os
import shutil
import scipy.misc
import random
try:
    import tensorflow as tf
except Exception as e:
    import warnings
    warnings.warn("need to install tensorflow to process event files in python. Message: {}".format(e))

def save_images_from_event(fn, tag, output_dir='./'):   # from monodepth2 repo
    assert(os.path.isdir(output_dir))

    image_str = tf.placeholder(tf.string)
    im_tf = tf.image.decode_image(image_str)

    sess = tf.InteractiveSession()
    with sess.as_default():
        count = 0
        for e in tf.train.summary_iterator(fn):
            for v in e.summary.value:
                if v.tag == tag:
                    im = im_tf.eval({image_str: v.image.encoded_image_string})
                    output_fn = os.path.realpath('{}/image_{:05d}.png'.format(output_dir, count))
                    print("Saving '{}'".format(output_fn))
                    scipy.misc.imsave(output_fn, im)
                    count += 1  

def event_fpath_from_dir(path_dir):
    fname = [p for p in os.listdir(path_dir) if "tfevents" in p]
    assert len(fname) == 1, fname
    path_in_file = os.path.join(path_dir, fname[0])
    return path_in_file

def event_reduce_image_given_tags(path_in, path_out, wanted_image_tags):
    path_in_file = event_fpath_from_dir(path_in)

    writer = tf.summary.FileWriter(path_out)
    for event in tf.train.summary_iterator(path_in_file):
        event_type = event.WhichOneof('what')
        if event_type != "summary":
            writer.add_event(event)
        else:
            wall_time = event.wall_time
            step = event.step
            # wanted_image_tags = ["rgb_00064/image", "rgb_00225/image", "rgb_00070/image"]
            condition_to_save = lambda v: (not v.HasField('image')) or (v.HasField('image') and v.tag in wanted_image_tags)
            filtered_value = [v for v in event.summary.value if condition_to_save(v)]
            if len(filtered_value) == 0:
                continue
            summary_new = tf.summary.Summary(value=filtered_value)
            event_new = tf.summary.Event(summary=summary_new, wall_time=wall_time, step=step)
            writer.add_event(event_new)
            
            # for value in event.summary.value:
            #     print(value.tag)
                # if (value.HasField('simple_value')):
                #     print(value.simple_value)
                #     summary.value.add(tag='{}'.format(value.tag),simple_value=value.simple_value)

    writer.flush()
    writer.close()
    return

def event_extract_tags(path, type_of_interest):
    """Here the path is to the directory, because the name of event file is determined by tensorboard summarywriter (instead of by us) when generated. 
    """
    path_file = event_fpath_from_dir(path)

    assert type_of_interest in ["simple_value", "image", "histo", "tensor", "audio"], type_of_interest

    tags = []
    for event in tf.train.summary_iterator(path_file):
        for value in event.summary.value:
            if value.WhichOneof('value') == type_of_interest:
                if value.tag not in tags:
                    tags.append(value.tag)
    return tags    

def event_reduce_image(path_in, path_out, n_tags_wanted):
    tags = event_extract_tags(path_in, "image")

    # tags_subset = random.sample(tags, 10)
    tags_subset = tags[:n_tags_wanted]

    event_reduce_image_given_tags(path_in, path_out, tags_subset)
    return

def event_view(path):
    """Here the path is to the directory, because the name of event file is determined by tensorboard summarywriter (instead of by us) when generated. 
    """
    path_file = event_fpath_from_dir(path)
    
    for event in tf.train.summary_iterator(path_file):
        for value in event.summary.value:
            print(event.step, value.tag, value.WhichOneof('value'))
    return

def event_merge(paths_in, path_out):
    writer = tf.summary.FileWriter(path_out)

    for path in paths_in:
        path_file = event_fpath_from_dir(path)

        for event in tf.train.summary_iterator(path_file):
            writer.add_event(event)

    writer.flush()
    writer.close()
    return

def event_modify_tags(path_in, path_out, type_of_interest, newtag_format):

    path_file = event_fpath_from_dir(path_in)
    writer = tf.summary.FileWriter(path_out)

    counter_dict = {}
    for event in tf.train.summary_iterator(path_file):
        wall_time = event.wall_time
        step = event.step

        if step not in counter_dict:
            counter_dict[step] = 0

        values_new = []
        for value in event.summary.value:
            if value.WhichOneof('value') == type_of_interest:
                new_tag = newtag_format.format(counter_dict[step], value.tag)
                value.tag = new_tag
                counter_dict[step] += 1
            values_new.append(value)
        
        summary_new = tf.summary.Summary(value=values_new)

        event_new = tf.summary.Event(summary=summary_new,
                                          wall_time=wall_time,
                                          step=step)
        writer.add_event(event_new)

    writer.flush()
    writer.close()
    return

if __name__ == "__main__":
    # ###### example for running save_images_from_event
    # path_to_event_file = '/home/minghanz/tmp/mono_model_KITTI_Lidar/val_Tue Dec  3 23:48:28 2019/events.out.tfevents.1575434908.MCity-GPU-Server'
    # tag_name = 'disp_0/gt_0'
    # output_dir = '/mnt/storage8t/minghanz/monodepth2_tmp/img_from_event'
    # save_images_from_event(path_to_event_file, tag_name, output_dir)

    #### view the tags in an event file
    # path_to_event_file = "/home/minghanz/DORN_pytorch/snap_dir/monodepth/Kitti/dorn/2021_03_02_16_16_45/tensorboard"
    # event_view(path_to_event_file)

    #### reduce the number of images in an event file
    # path_to_event_file = "/home/minghanz/DORN_pytorch/snap_dir/monodepth/Kitti/dorn/2021_03_09_22_54_11/tensorboard"
    # path_to_event_file_new = "/home/minghanz/DORN_pytorch/snap_dir/monodepth/Kitti/dorn_test/2021_03_09_22_54_11/tensorboard"
    # event_reduce_image(path_to_event_file, path_to_event_file_new, n_tags_wanted=10)
    # event_view(path_to_event_file_new)
    
    #### merge multiple event files with non-overlapping tags to a single event file (actually merging to a single folder is necessary)
    # root = "/home/minghanz/DORN_pytorch/snap_dir/monodepth/Kitti/dorn/2021_03_02_16_16_45/tensorboard"
    # folders = os.listdir(root) 
    # folders = [os.path.join(root, f) for f in folders]
    # folders = [p for p in folders if os.path.isdir(p)]
    # folders.append(root)
    # path_out = "/home/minghanz/DORN_pytorch/snap_dir/monodepth/Kitti/dorn/2021_03_02_16_16_45/tensorboard_c"
    # event_merge(folders, path_out)
    # event_view(path_out)
    
    #### renaming the tags
    root = "/home/minghanz/DORN_pytorch/snap_dir/monodepth/vKitti2/dorn"
    folders = os.listdir(root)
    for folder in folders:
        path_in = os.path.join(root, folder, "tensorboard")
        path_out = os.path.join(root, folder, "tensorboard_r")
        event_modify_tags(path_in, path_out, "image", "{:02d}_{}")
        shutil.rmtree(path_in)
        os.rename(path_out, path_in)