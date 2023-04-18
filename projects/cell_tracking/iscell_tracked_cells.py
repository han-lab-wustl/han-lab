# https://github.com/MouseLand/suite2p/issues/292
import numpy as np, os
from scipy import stats
from scipy.io import loadmat
import h5py

# make stat array
fpath = r'X:\sstcre_imaging\e201\week6\suite2p\plane0'
iscell=np.load(os.path.join(fpath, 'iscell.npy'),
allow_pickle=True)

tracked_cells = r'Y:\sstcre_analysis\celltrack\e201_week5-8\Results\commoncells.mat'
f = h5py.File(tracked_cells, 'r')
tracked_cells_arr = f['commoncells'][:].astype(int)

cellind = np.arange(0,len(iscell))
iscell_v2 = [[int(xx in tracked_cells_arr.T[:,1]) ,iscell[i,1]] for i,
             xx in enumerate(cellind)]
np.save(os.path.join(fpath,'iscell.npy'), np.array(iscell_v2))