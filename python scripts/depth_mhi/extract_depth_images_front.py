import matplotlib.pyplot as plt
import sys

plt.rcParams.update({'figure.max_open_warning': 0})

cx = 255.165695200749 # main point, the center of the image
cy = 211.824600345805
fx = 367.286994337726 # focal length in the direction of x and y
fy = 367.286855347968

def depth_to_point_cloud_pos(x, y, d):
    pz = d
    px = (x - cx) * pz / fx
    py = (y - cy) * pz / fy
    return (px, py, pz)

def main():
    fname = sys.argv[1]
    d = []
    with open(fname) as f:
        first_line = f.readline()
        parts = [int(num) for num in first_line.split(',')]
        nframes = parts[0]
        nrows = parts[1]
        ncols = parts[2]

        for k in range(nframes):
            depth_map = [f.readline() for i in range(nrows)]
            l = [[int(num) for num in line.split(',')] for line in depth_map]
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
                (xs, ys, zs) = depth_to_point_cloud_pos(i, j, d[k][i][j])
                if ys >= -2000 and ys <= 1000 and xs >= -2000 and xs <= 0 and \
                   zs < 2300:
                    frame_xs.append(xs)
                    frame_ys.append(ys)
                    frame_zs.append(zs)      
        Xs.append(frame_xs)
        Ys.append(frame_ys)
        Zs.append(frame_zs)

    # ----- Plot projections -----
    for frame_id in range(len(Ys)):
        fig = plt.figure()#frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        ax.set_xlim(-2000, 1000)
        ax.set_ylim(0, 2000)
        fig.add_axes(ax)

        plt.scatter(Ys[frame_id], [-x for x in Xs[frame_id]], c='black', s=0.3)
        plt.grid(False)
        plt.savefig(str(frame_id) + '_front' + '.jpg')

if __name__ == "__main__":
    main()
