import os
import math

# Byte order: little/big endian.
bo = "little"

def extract_depth_data(fname, parent_dir, child_dir, work_dir):
    with open(parent_dir + child_dir + '/' + fname, "rb") as f:
        nframes = int.from_bytes(f.read(4), bo)
        ncols = int.from_bytes(f.read(4), bo)
        nrows = int.from_bytes(f.read(4), bo)

        filename = fname[:-4] + ".txt"
        with open(work_dir + child_dir + '/' + filename, "w") as g:
            g.write(str(nframes) + ',' + str(nrows) + ',' + str(ncols) + '\n')
            
            for frameId in range(nframes):
                for i in range(nrows):
                    # Read and write depth data.
                    for j in range(ncols):
                        if j != 0:
                            g.write(',')
                        g.write(str(int.from_bytes(f.read(4), bo)))
                    g.write('\n')

                    # Simply skip Kinect skeleton data.
                    f.read(ncols)

def main():
    parent_dir = '/media/ana-cosmina/Windows8_OS/Users/Comy/Downloads/' \
                 'MSRDailyActivity3D/Depth/'
    work_dir = '/home/ana-cosmina/Desktop/depth_data/'

    for child_dir in os.listdir(parent_dir):
        if child_dir == 'eat': 
            for fname in os.listdir(parent_dir + child_dir):
                if 'a02_s09_e02' in fname:
                    extract_depth_data(fname, parent_dir, child_dir, work_dir)
                    print("Converted", fname)

if __name__ == "__main__":
    main()
