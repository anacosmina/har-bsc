import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

DEFAULT_THRESHOLD = 30

def compute_mhi(video_src):
    cam = cv.VideoCapture(video_src)
    if not cam.isOpened():
        print("could not open video_src " + str(video_src) + " !\n")

    cam.set(cv.CAP_PROP_POS_AVI_RATIO,1)
    MHI_DURATION = cam.get(cv.CAP_PROP_POS_MSEC)

    cam = cv.VideoCapture(video_src)
    if not cam.isOpened():
        print("could not open video_src " + str(video_src) + " !\n")

    ret, frame = cam.read()
    if ret == False:
        print("could not read from " + str(video_src) + " !\n")

    h, w = frame.shape[:2]
    prev_frame = frame.copy()
    motion_history = np.zeros((h, w), np.float32)

    while cam.isOpened():
        ret, frame = cam.read()
        if ret == False:
            break
        frame_diff = cv.absdiff(frame, prev_frame)
        gray_diff = cv.cvtColor(frame_diff, cv.COLOR_BGR2GRAY)
        ret, motion_mask = cv.threshold(gray_diff, DEFAULT_THRESHOLD, 1,        
                                        cv.THRESH_BINARY)
        timestamp = cv.getTickCount() / cv.getTickFrequency() * 1000
        cv.motempl.updateMotionHistory(motion_mask, motion_history, timestamp, 
                                       MHI_DURATION)
        vis = np.uint8(np.clip((motion_history-(timestamp-MHI_DURATION)) / \
                                MHI_DURATION, 0, 1) * 255)
        prev_frame = frame.copy()
    cam.release()
    
    return vis

def main():
    parent_dir = '/media/ana-cosmina/COMY/_DATASET cropped/RGB/'
    work_dir = '/home/ana-cosmina/Desktop/mhi_own/'

    for child_dir in os.listdir(parent_dir):
        os.mkdir(work_dir + child_dir)
        os.chdir(work_dir + child_dir)
        
        for fname in os.listdir(parent_dir + child_dir):
            video_src = parent_dir + child_dir + '/' + fname
            mhi = compute_mhi(video_src)

            fig = plt.figure(frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)

            ax.matshow(mhi, cmap=plt.cm.gray)
            plt.savefig(fname[:-4])

        print("Done for", child_dir)


if __name__ == "__main__":
    main()
