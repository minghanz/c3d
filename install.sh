#!/bin/sh
for subdir in custom_dense custom_dense_angle custom_dense_normal custom_dense_with_normal custom_cross_prod custom_cross_subtract custom_norm
do
    cd "$subdir"
    if [ -d build ]
    then
        echo "Previously built files exist in ${subdir}, removing them now."
        ##[-d build ] && rm build -r
        rm build -r
    fi
    python setup.py build_ext
    echo "Built executable in $subdir"
    if [ "$subdir" = custom_cross_prod ] || [ "$subdir" = custom_cross_subtract ] || [ "$subdir" = custom_norm ]
    then
        # echo "haha"
        cp -v build/*/*.so ../../pytorch-unet/ ## being verbose about what's going on
    else
        # echo "hoho"
        cp -v build/*/*.so ../../monodepth2/ ## being verbose about what's going on 
    fi
    cd ..
done