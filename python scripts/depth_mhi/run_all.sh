#!/bin/sh

EXT=txt
for i in *.${EXT}; do
    #python3 extract_depth_images_front.py $i
    #python3 images_to_video.py
    #mv video.avi ..
    #cd ..
    #python3 mhi.py $i 'front'
    #cd depth_mhi

    python3 extract_depth_images_side.py $i
    python3 images_to_video.py
    mv video.avi ..
    cd ..
    python3 mhi.py $i 'side'
    cd depth_mhi

    python3 extract_depth_images_top.py $i
    python3 images_to_video.py
    mv video.avi ..
    cd ..
    python3 mhi.py $i 'top'
    cd depth_mhi
done

