# zahra
# multi session cebra
import numpy as np
import scipy.io
import numpy as np, os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import cebra
from cebra import CEBRA
import glob

matplotlib.use('TkAgg') #might need for gui

# days = [67,70,73,76,80,83,86,89]
# days = np.arange(1,9)
days = [4,5,6,7]#
# src = r'Y:\sstcre_imaging\e200'
# src = r'Z:\cellreg1month_Fmats\E186'
src = r'F:\E145'
mat = []
for day in days:
#        mat.append(glob.glob(os.path.join(src, f'{day}', '**', '*Fall.mat'), recursive = True)[0])
    mat.append(glob.glob(os.path.join(src, f'Day{day}', '**', 'plane*', '*Fall.mat'), recursive = True))
mat = np.ravel(mat) # include planes as a separate session
dst = r'Y:\sstcre_analysis\hrz\cebra\saved_models'

dataz = []; posz = []
eps = []
# rewzone = 2
# ep corresponding to rew loc 3 in these days
# eps = [1, 2, 3, 1, 1, 3, 1, 2]
planes = 3
for i in np.arange(0,len(mat),planes):
        if planes > 1:
                for pl in range(planes):
                        neural_data = cebra.load_data(file=mat[i+pl], 
                                        key="spks")
                        iscell = cebra.load_data(file=mat[i+pl], 
                                key="iscell")
                        trials = np.squeeze(cebra.load_data(file=mat[i+pl], 
                                        key="trialnum"))
                        changeRewLoc = cebra.load_data(file=mat[i+pl], 
                                        key="changeRewLoc")
                        ybinned = cebra.load_data(file=mat[i+pl], 
                                        key="ybinned")
                        rews = cebra.load_data(file=mat[i+pl], 
                                key="rewards")
                        
                        # epoch 1 only
                        neural_data = neural_data[iscell[:,0].astype(bool)]
                        rewlocs = changeRewLoc[changeRewLoc>0]
                        ep = np.where(rewlocs>=135)[0][0]+1 # pick epoch with reward location 3
                        eps.append(ep)
                #rewzones
                #rewloc>=135 = 3
                #(rewloc>100) & (rewloc<121) = 2
                #rewloc<=86
                # remember indexing is from 0
                #     for ep, rewloc in enumerate(rewlocs): if doing all epochs
                        # ep = ep+1
                        epstart = np.where(changeRewLoc>0)[1][ep-1]
                        try:
                                epend = np.where(changeRewLoc>0)[1][ep]
                        except Exception as e:
                                epend = len(np.squeeze(changeRewLoc)) # assumes animal did not start ep4, so just get end of recording
                                print(e)
                        rewloc = changeRewLoc[changeRewLoc>0][ep-1] # rew location in position
                        # for entire session
                        # epstart = 0; epend = len(np.squeeze(changeRewLoc))
                        mask = np.arange(epstart, epend) # only specified epoch
                        pos = np.squeeze(ybinned)[mask] 
                        pos_ = pos[pos>3] # only position after dark time
                        mask = mask[pos>3]
                        trials_ = trials[mask]
                        rews_ = np.squeeze(rews)[mask]
                        success_trials = [] # does not matter if only looking at probes
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
                        
                        if data.shape[0]>0:
                                dataz.append(data)
                                posz.append(pos_)
                        else:
                                print(f'\n day {i+1} not used, not enuf successful trials')

print(len(dataz))
#%%
multi_cebra_model  = cebra.CEBRA(batch_size=512,
                                 output_dimension=3,
                                 max_iterations=15000,
                                 temperature_mode="auto",
                                 max_adapt_iterations=500)
multi_cebra_model.fit(dataz, posz)
multi_cebra_model.save(os.path.join(dst, f'dim3_e145_day4_rewzone3_last8trials_successtrials_before_rewloc.pt'))

# for session in range(len(mat)):
# multi_cebra_model = cebra.CEBRA.load(os.path.join(dst, f'dim3_e145_days1-12_rewzone3_last8trials_successtrials_before_rewloc.pt'))
session = 0
# import data for this model (rerun top part)
embedding = multi_cebra_model.transform(dataz[session], session_id=session)
pos = posz[session]
cebra.plot_embedding(embedding, pos, cmap = plt.cm.cool, markersize=5)
norm = matplotlib.colors.Normalize(min(pos), max(pos))
cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=plt.cm.cool))
cbar.set_label('ybinned', rotation=270)
plt.title(f"Day {session}")#, epoch = {eps[session]}")
plt.ion()

cebra.plot_loss(multi_cebra_model)