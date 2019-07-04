from math import sqrt
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os

parent_dir = '/home/ana-cosmina/Desktop/depth_data/'
work_dir = '/home/ana-cosmina/Desktop/dmm/'

cx = 255.165695200749 # main point, the center of the image
cy = 211.824600345805
fx = 367.286994337726 # focal length in the direction of x and y
fy = 367.286855347968

NORM_CT = 255

def depth_to_point_cloud_pos(x, y, d):
    pz = d
    px = (x - cx) * pz / fx
    py = (y - cy) * pz / fy
    return (px, py, pz)

def max_value(lst):
    return max([max(sublst) for sublst in lst])

def min_value(lst):
    return min([min(sublst) for sublst in lst])

def shift_over_zero(lst):
    min_val = min_value(lst)
    if min_val < 0:
        offset = -min_val
        return [[x + offset for x in sublst] for sublst in lst]
    return lst

def normalize(mat, max_val):
    nrows = len(mat)
    ncols = len(mat[0])
    nmat = [[0 for j in range(ncols)] for i in range(nrows)]

    for i in range(nrows):
        for j in range(ncols):
            nmat[i][j] = round(mat[i][j] * NORM_CT / max_val)
            
    return nmat

def compute_map(A1, A2, nframes):
    my_map = []
    for k in range(nframes):
        frame_map = [[0] * (NORM_CT + 1) for _ in range(NORM_CT + 1)]
        for a1, a2 in zip(A1[k], A2[k]):
            frame_map[a1][a2] = 1
        my_map.append(frame_map)
    return my_map

def get_dmm(map_f, nframes):
    dmm = [[0 for j in range(NORM_CT + 1)] for i in range(NORM_CT + 1)]
    
    for k in range(1, nframes):
        old_map = map_f[k - 1]
        cur_map = map_f[k]
        for i in range(NORM_CT + 1):
            for j in range(NORM_CT + 1): 
                if cur_map[i][j] != old_map[i][j]:
                    dmm[i][j] += 1
    return dmm

def dmm_to_png(dmm, fname, dmm_type):
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.matshow(dmm, cmap=plt.cm.binary)
    plt.savefig(fname[:-4] + '_' + dmm_type + '.png')


def main():
    for child_dir in os.listdir(parent_dir):
        os.chdir(work_dir + child_dir)

        for fname in os.listdir(parent_dir + child_dir):
            print("Started processing", fname)
            filename = parent_dir + child_dir + '/' + fname
            d = []
            with open(filename) as f:
                first_line = f.readline()
                parts = [int(num) for num in first_line.split(',')]
                nframes = parts[0]
                nrows = parts[1]
                ncols = parts[2]

                for k in range(nframes):
                    depth_map = [f.readline() for i in range(nrows)]
                    l = [[int(num) for num in line.split(',')] \
                         for line in depth_map]
                    d.append(l)

            Xs = []
            Ys = []
            Zs = []
            for k in range(nframes):
                frame_xs = []
                frame_ys = []
                frame_zs = []
                for i in range(nrows):
                    for j in range(ncols):
                        (xs, ys, zs) = depth_to_point_cloud_pos(i, j,
                                                                d[k][i][j])
                        frame_xs.append(xs)
                        frame_ys.append(ys)
                        frame_zs.append(zs)
                Xs.append(frame_xs)
                Ys.append(frame_ys)
                Zs.append(frame_zs)

            Xs = shift_over_zero(Xs)
            Ys = shift_over_zero(Ys)
            Zs = shift_over_zero(Zs)
            
            # Normalize all values between 0 and NORM_CT.
            max_val = max([max_value(lst) for lst in [Xs, Ys, Zs]])
            Xs = normalize(Xs, max_val)
            Ys = normalize(Ys, max_val)
            Zs = normalize(Zs, max_val)

            map_f = compute_map(Xs, Ys, nframes)
            map_s = compute_map(Xs, Zs, nframes)
            map_t = compute_map(Ys, Zs, nframes)

            dmm_to_png(get_dmm(map_f, nframes), fname, 'front')
            dmm_to_png(get_dmm(map_s, nframes), fname, 'side')
            dmm_to_png(get_dmm(map_t, nframes), fname, 'top')
            break


if __name__ == "__main__":
    main()
