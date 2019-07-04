import cv2 as cv
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import re

# Byte order: little/big endian.
bo = "little"

DEFAULT_THRESHOLD = 100

def extract_depth_data(fname, parent_dir, child_dir, work_dir):
    d = []
    with open(parent_dir + child_dir + '/' + fname, "rb") as f:
        nframes = int.from_bytes(f.read(4), bo)
        ncols = int.from_bytes(f.read(4), bo)
        nrows = int.from_bytes(f.read(4), bo)

        for frameId in range(nframes):
            l_out = []
            for i in range(nrows):
                l_in = []
                # Read and write depth data.
                for j in range(ncols):
                    l_in.append(int.from_bytes(f.read(4), bo))
                # Simply skip Kinect skeleton data.
                f.read(ncols)
                l_out.append(l_in)
            d.append(l_out)
    return d


def create_images(d):
    # ----- Plot projections -----
    for frame_id in range(len(d)):
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        ax.matshow(d[frame_id], cmap=plt.cm.gist_gray)
        plt.grid(False)
        plt.savefig(str(frame_id) + '.png')

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def create_video():
    image_folder = '.'
    video_name = 'video.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=natural_keys)
    frame = cv.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv.VideoWriter(video_name, cv.VideoWriter_fourcc('M','J','P','G'),
                            30, (width,height))

    for image in images:
        video.write(cv.imread(os.path.join(image_folder, image)))

    cv.destroyAllWindows()
    video.release()

    for image in images:
        os.remove(image)

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

def save_mhi(mhi, fname):
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.matshow(mhi, cmap=plt.cm.gray)
    plt.savefig(fname)

def main():
    parent_dir = '/media/ana-cosmina/Windows8_OS/Users/Comy/Downloads/' \
                 'MSRDailyActivity3D/Depth/'
    work_dir = '/home/ana-cosmina/Desktop/depth_images/'

    for child_dir in os.listdir(parent_dir):
        os.mkdir(work_dir + child_dir)
        os.chdir(work_dir + child_dir)

        for fname in os.listdir(parent_dir + child_dir):
            d = extract_depth_data(fname, parent_dir, child_dir, work_dir)
            create_images(d)
            create_video()
            #mhi = compute_mhi('video.avi')
            #save_mhi(mhi, fname[:-4] + '.jpg')
            #os.remove('video.avi')
            break

if __name__ == "__main__":
    main()
