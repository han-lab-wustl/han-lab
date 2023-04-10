# https://github.com/MouseLand/suite2p/issues/292
import numpy as np, os
from scipy import stats
from scipy.io import loadmat

# make stat array
fpath = r'D:\suite2p_param_sweep\week6\suite2p\plane0'
stat=np.load(os.path.join(fpath, 'stat.npy'), allow_pickle=True)
iscell=np.load(os.path.join(fpath, 'iscell_old.npy'),
allow_pickle=True)

tracked_cells = r'Y:\sstcre_analysis\celltrack\e201_week1-6\Results\commoncells_TEST_for_suite2p_iscell.mat'
tracked_cells=loadmat(tracked_cells)["commoncells"]
cellind = np.arange(0,len(iscell))
iscell_v2 = [[int(xx in tracked_cells[:,5]) ,iscell[i,1]] for i,xx in enumerate(cellind)]
np.save(os.path.join(fpath,'iscell.npy'), np.array(iscell_v2))