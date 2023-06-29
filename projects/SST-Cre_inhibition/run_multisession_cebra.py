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

days = np.arange(1,9)#np.arange(1,9)#[67,70,73,76,80,83,86,89]
src = r'Z:\cellreg1month_Fmats\E186'#r'Y:\sstcre_imaging\e200'
mat = []
for day in days:       
    #    mat.append(glob.glob(os.path.join(src, f'{day}', '**', '*Fall.mat'), recursive = True)[0])
    mat.append(glob.glob(os.path.join(src, f'd{day:03d}', '**', '*Fall.mat'), recursive = True)[0])

dst = r'Y:\sstcre_analysis\hrz\cebra\saved_models'

dataz = []; posz = []
eps = []
# rewzone = 2
# ep corresponding to rew loc 3 in these days
# eps = [1, 2, 3, 1, 1, 3, 1, 2]

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
    mask = np.arange(epstart, epend) # only specified epoch
    pos = np.squeeze(ybinned)[mask] 
    pos_ = pos[pos>1] # only position after dark time
    mask = mask[pos>1]
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
    # single model training to compare
    # cebra time + behavior
    # single_cebra_model_pos = cebra.CEBRA(batch_size=512,
    #                                 output_dimension=3,
    #                                 max_iterations=15000,
    #                                 temperature_mode="auto",
    #                                 max_adapt_iterations=500)
    # single_cebra_model_pos.fit(data, pos_)
    # embedding = single_cebra_model_pos.transform(data)
    # # discrete var
    # cebra.plot_embedding(embedding, pos_, cmap = plt.cm.cool)
    # norm = matplotlib.colors.Normalize(min(pos_), max(pos_))
    # cbar = plt.colorbar(cm.ScalarMappable(norm=norm, 
    #         cmap=plt.cm.cool))
    # cbar.set_label('ybinned', rotation=270)
    # plt.title(f"Rew loc = {rewloc}\n Epoch = {ep}")
    # plt.ion()
    # single_cebra_model_pos.save(f'Y:\\sstcre_analysis\\hrz\\cebra\\saved_models\\dim3_e200_ctrlday{days[i]}_rewzone{rewzone}_last8trials_successtrials_before_rewloc.pt')
    if data.shape[0]>0:
        dataz.append(data)
        posz.append(pos_)
    else:
          print(f'\n day {i+1} not used, not enuf successful trials')

multi_cebra_model  = cebra.CEBRA(batch_size=512,
                                 output_dimension=3,
                                 max_iterations=15000,
                                 temperature_mode="auto",
                                 max_adapt_iterations=500)
multi_cebra_model.fit(dataz, posz)
multi_cebra_model.save(os.path.join(dst, f'dim3_e186_days1-8_rewzone3_last8trials_successtrials_before_rewloc.pt'))

# for session in range(len(mat)):
# multi_cebra_model = cebra.CEBRA.load(os.path.join(dst, f'dim3_e200_ctrlutilday89_ep2_first3trials_successtrials_before_rewloc.pt'))
session = 2
# import data for this model (rerun top part)
embedding = multi_cebra_model.transform(dataz[session], session_id=session)
pos = posz[session]
cebra.plot_embedding(embedding, pos, cmap = plt.cm.cool, markersize=5)
norm = matplotlib.colors.Normalize(min(pos), max(pos))
cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=plt.cm.cool))
cbar.set_label('ybinned', rotation=270)
plt.title(f"Day {session}, epoch = {eps[session]}")
plt.ion()

    # cebra.plot_loss(multi_cebra_model)