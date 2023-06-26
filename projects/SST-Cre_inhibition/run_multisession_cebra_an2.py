# zahra
# multi session cebra
import numpy as np, glob
import scipy.io
import numpy as np, os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import cebra
from cebra import CEBRA

matplotlib.use('TkAgg') #might need for gui


days = [57, 60, 63, 66, 69, 72, 76, 79, 85]
#[56, 59, 62, 65, 68, 71, 75, 78, 82, 84, 87, 89]
src = r'Z:\sstcre_imaging\e201'
mat = []
for day in days:       
    mat.append(glob.glob(os.path.join(src, str(day), '**', '*Fall.mat'), recursive = True)[0])

dst = r'Y:\sstcre_analysis\hrz\cebra\saved_models'

# Load the .npz
dataz = []; posz = []; 
# ep corresponding to rew loc 3 in these days
eps = []#[3, 3, 3, 1, 3, 2, 1, 3, 3]
for i in range(len(mat)):
    
    neural_data = cebra.load_data(file=mat[i], 
                key="spks")
    iscell = cebra.load_data(file=mat[i], 
            key="iscell")
    trials = np.squeeze(cebra.load_data(file=mat[i], 
                key="trialnum"))
    changeRewLoc = cebra.load_data(file=mat[i], 
                key="changeRewLoc")
    ybinned = cebra.load_data(file=mat[i], 
                key="ybinned")
    rews = cebra.load_data(file=mat[i], 
            key="rewards")
    
    # epoch 1 only
    neural_data = neural_data[iscell[:,0].astype(bool)]
    rewloc = changeRewLoc[changeRewLoc>0]
    ep = np.where(rewloc>=135)[0][0]+1 # pick epoch with reward location 3
    eps.append(ep)
    #rewzones
    #rewloc>=135 = 3
    #(rewloc>100) & (rewloc<121) = 2
    #rewloc<=86
    # remember indexing is from 0
    epstart = np.where(changeRewLoc>0)[1][ep-1]
    try:
            epend = np.where(changeRewLoc>0)[1][ep]
    except Exception as e:
            epend = len(np.squeeze(changeRewLoc)) # assumes animal did not start ep4, so just get end of recording
            print(e)
    rewloc = changeRewLoc[changeRewLoc>0][ep-1] # rew location in position
    # for entire session
    # epstart = 0; epend = len(np.squeeze(changeRewLoc))
    mask = np.arange(epstart, epend)
    pos = np.squeeze(ybinned)[mask]
    pos_ = pos[pos>1]
    mask = mask[pos>1] 
    trials_ = trials[mask]
    rews_ = np.squeeze(rews)[mask]
    success_trials = []
    for tr in np.unique(trials_):
            if sum(rews_[trials_ == tr])>0:
                    success_trials.append(tr)
    # only successful trials - shouldnt make a huge difference for e201
    # get only ybinned before or after rew loc (-20cm from middle of rew zone)
    # or 20 cm around rew loc
    dist_rewloc = 20
    # mask2 = (trials[mask]>=3) & (pos_<(rewloc-dist_rewloc))
    # first 5 trials: (trials_>=3) & (trials_<8) 
    # last 8 trials: trials_>=max(trials_)-8
    # probes: trials_<3
    # mask2 = (np.in1d(trials_, np.array(success_trials))) & (trials_<3) & (pos_<(rewloc-dist_rewloc))
    mask2 = (np.in1d(trials_, np.array(success_trials))) & (trials_>=max(trials_)-8) & (pos_<(rewloc-dist_rewloc))
    #& (pos_<=(rewloc+dist_rewloc)) 
    pos_ = pos_[mask2]
    # IMPORTANT! EXCLUDE 'DARK TIME' (YBINNED 1 IN ENTIRE DARK TIME)
    data = neural_data.T[mask][mask2]
    dataz.append(data)
    posz.append(pos_)

multi_cebra_model  = cebra.CEBRA(batch_size=512,
                                 output_dimension=3,
                                 max_iterations=15000,
                                 temperature_mode='auto',
                                 max_adapt_iterations=500)
multi_cebra_model.fit(dataz, posz)
multi_cebra_model.save(os.path.join(dst, f'dim3_e201_ctrlutilday85_rewzone3_last8trials_successtrials_before_rewloc.pt'))

session = 2
embedding = multi_cebra_model.transform(dataz[session], session_id=session)
pos = posz[session]
cebra.plot_embedding(embedding, pos, cmap = plt.cm.cool, markersize = 5)
norm = matplotlib.colors.Normalize(min(pos), max(pos))
cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=plt.cm.cool))
cbar.set_label('ybinned', rotation=270)
plt.title(f"Day {session+1}, \n Epoch = {eps[session]}")
plt.ion()

# cebra.plot_loss(multi_cebra_model)