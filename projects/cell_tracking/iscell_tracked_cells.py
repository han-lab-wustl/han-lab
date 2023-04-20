# https://github.com/MouseLand/suite2p/issues/292
import numpy as np, os
from scipy import stats
from scipy.io import loadmat
import h5py

# make stat array
tracked_cells = r'Y:\sstcre_analysis\celltrack\e201_week5-8_stringent\Results\commoncells.mat'
tracked_cells_arr = loadmat(tracked_cells)['commoncells']

weeks = ['week5', 'week6', 'week7', 'week8']
for i,week in enumerate(weeks): 
    fpath = rf'X:\sstcre_imaging\e201\{week}\suite2p\plane0'
    iscell=np.load(os.path.join(fpath, 'iscell.npy'),
    allow_pickle=True)

    cellind = np.arange(0,len(iscell))
    iscell_v2 = [[int(xx in tracked_cells_arr[:,i]) ,iscell[ii,1]] for ii,
                xx in enumerate(cellind)]
    np.save(os.path.join(fpath,'iscell.npy'), np.array(iscell_v2))
    print(week)

tracked_cells = r'Y:\sstcre_analysis\celltrack\e201_week5-8_permissive\Results\commoncells.mat'
tracked_cells_arr = loadmat(tracked_cells)['commoncells']

weeks = ['week5_permissive', 'week6_permissive', 'week7_permissive', 'week8_permissive']
for i,week in enumerate(weeks): 
    fpath = rf'X:\sstcre_imaging\e201\{week}\suite2p\plane0'
    iscell=np.load(os.path.join(fpath, 'iscell.npy'),
    allow_pickle=True)

    cellind = np.arange(0,len(iscell))
    iscell_v2 = [[int(xx in tracked_cells_arr[:,i]) ,iscell[ii,1]] for ii,
                xx in enumerate(cellind)]
    np.save(os.path.join(fpath,'iscell.npy'), np.array(iscell_v2))
