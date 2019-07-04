import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    work_dir = '/home/ana-cosmina/Desktop/own_mhi_grouped/'

    for child_dir in os.listdir(work_dir):
        os.chdir(work_dir + child_dir)
        
        for fname in os.listdir(work_dir + child_dir):
            im_src = work_dir + child_dir + '/' + fname
            flipped = np.fliplr(plt.imread(im_src))

            fig = plt.figure(frameon=False, figsize=(640, 480), dpi=1)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)

            ax.imshow(flipped)
            plt.savefig(fname[:-4] + '_flipped.png')

        print("Done for", child_dir)


if __name__ == "__main__":
    main()
