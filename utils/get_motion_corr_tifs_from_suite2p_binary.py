# Zahra
# as the title says
# needs suite2p installed
import suite2p
import tifffile, numpy as np, os
pth = 'Z:\sstcre_imaging\e201\week6\suite2p\plane0\data.bin'
Ly = 512
Lx = 629 # from ops
f_input2 = suite2p.io.BinaryRWFile(Ly=Ly, Lx=Lx, filename=pth)
dst = 'X:\\week6_e201_motion_corrected'
for i in range(0,f_input2.shape[0],1000):
    print(i)
    tifffile.imwrite(os.path.join(dst, f'file_{i:08d}.tif'), f_input2[i:i+1000])