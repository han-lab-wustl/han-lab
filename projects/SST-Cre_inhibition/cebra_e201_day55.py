import scipy.io
import numpy as np, os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
matplotlib.use('TkAgg') #might need for gui

# mat = r'Z:\cellreg1month_Fmats\E186\d009\221215_YC_Fall.mat'
# mat = r'Z:\sstcre_imaging\e201\56\230503_ZD_000_001\suite2p\plane0\Fall.mat'
mat = r'Y:\sstcre_analysis\hrz\cebra\open_datasets\tests_hc30.mat'
# only use epoch 1?
dst = r'Y:\sstcre_analysis\hrz\cebra\saved_models'
import cebra
from cebra import CEBRA

neural_data = cebra.load_data(file=mat, key='traces1')
# Load the .npz
neural_data_og = cebra.load_data(file=mat, 
            key="spks")
iscell = cebra.load_data(file=mat, 
            key="iscell")
# iscell = cebra.load_data(file=iscellpth)
trials = cebra.load_data(file=mat, 
            key="trialnum")
changeRewLoc = cebra.load_data(file=mat, 
            key="changeRewLoc")
rews = cebra.load_data(file=mat, 
            key="rewards")
ybinned = cebra.load_data(file=mat, 
            key="ybinned")
velocity = cebra.load_data(file=mat, 
            key="forwardvel")
trials = np.squeeze(trials)

neural_data = neural_data_og[iscell[:,0].astype(bool)]
#%%
#vs. ep2 + exclude probes
# SET EPOCH
# eps=[1, 2, 3]
# for ep in eps: 
ep = 2
# define epochs
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
pos_ = pos[pos>1] # 1 in new HRZ, 3 in old to exclude dark time
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
mask2 = (np.in1d(trials_, np.array(success_trials))) & (trials_>=max(trials_)-8) & (pos_<(rewloc-dist_rewloc))
#& (pos_<=(rewloc+dist_rewloc)) 
pos_ = pos_[mask2]
# IMPORTANT! EXCLUDE 'DARK TIME' (YBINNED 1 IN ENTIRE DARK TIME)
data = neural_data.T[mask][mask2]# & (trials[mask]<8)][pos>1]
# data = neural_data.T[trials>=3]
# vel = np.squeeze(velocity)[mask][trials[mask]>=3]
# discrete var
# rew = np.squeeze(rews)[mask][trials[mask]>=3]==1 # ignore cs
# pos = np.squeeze(ybinned)[trials>=3]

timesteps = data.shape[0]
neurons = data.shape[1]
out_dim = 3

# cebra time
# single_cebra_model = cebra.CEBRA(batch_size=512,
#                                  output_dimension=out_dim,
#                                  max_iterations=15000,
#                                  temperature_mode="auto",
#                                  max_adapt_iterations=10)
# single_cebra_model.fit(data)
# # temperature auto = iterates thru temperature parameter
# embedding = single_cebra_model.transform(data)
# # assert(embedding.shape == (timesteps, out_dim))
# cebra.plot_embedding(embedding)
# cebra.plot_loss(single_cebra_model)
# cebra.plot_temperature(single_cebra_model)

# cebra time + behavior
single_cebra_model_pos = cebra.CEBRA(batch_size=512,
                                output_dimension=out_dim,
                                max_iterations=15000,
                                temperature_mode="auto",
                                max_adapt_iterations=500)
single_cebra_model_pos.fit(data, pos_)
embedding = single_cebra_model_pos.transform(data)
# discrete var
cebra.plot_embedding(embedding, pos_, cmap = plt.cm.cool, markersize=5)
norm = matplotlib.colors.Normalize(min(pos_), max(pos_))
cbar = plt.colorbar(cm.ScalarMappable(norm=norm, 
        cmap=plt.cm.cool))
cbar.set_label('ybinned', rotation=270)
plt.title(f"Rew loc = {rewloc}\n Epoch = {ep}")
plt.ion()
single_cebra_model_pos.save(os.path.join(dst, f'dim3_e186_d9_ep2_last8trials_successtrials_before_rewloc.pt'))

# check other latents
# fig = plt.figure(figsize=(10,5))
# ax1 = fig.add_subplot(131, projection="3d")
# ax2 = fig.add_subplot(132, projection="3d")
# ax3 = fig.add_subplot(133, projection="3d")     

# ax1 = cebra.plot_embedding(embedding, embedding_labels=pos_, 
# idx_order=(0,1,2), title="Latents: (1,2,3)", ax=ax1)
# ax2 = cebra.plot_embedding(embedding, embedding_labels=pos_, 
# idx_order=(3,4,5), title="Latents: (4,5,6)", ax=ax2)
# ax3 = cebra.plot_embedding(embedding, embedding_labels=pos_, 
# idx_order=(5,6,7), title="Latents: (6,7,8)", ax=ax3)

cebra.plot_loss(single_cebra_model_pos)
cebra.plot_temperature(single_cebra_model_pos)

# single_cebra_model_pos.save('Y:\\sstcre_analysis\\hrz\\cebra\\saved_models\\dim3_ep4_e201_day56_first5trials_before_rew_loc.pt')
#%%
# apply ep4 model to ep1 (before rew loc)
# define epochs
ep=1
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
# only last 8 trials
mask2 = (np.in1d(trials_, np.array(success_trials))) & (trials_>=max(trials_)-8) & (pos_<(rewloc-dist_rewloc))
#& (pos_<=(rewloc+dist_rewloc)) 
pos_ = pos_[mask2]
# IMPORTANT! EXCLUDE 'DARK TIME' (YBINNED 1 IN ENTIRE DARK TIME)
data2 = neural_data.T[mask][mask2]
# get loss score
single_score = cebra.sklearn.metrics.infonce_loss(single_cebra_model_pos,
                                                  data2,
                                                  pos_,
                                                  num_batches=5)
# adapt model
single_cebra_model_pos.fit(data2, pos_, adapt=True)
embedding = single_cebra_model_pos.transform(data2)
cebra.plot_embedding(embedding, pos_, cmap = plt.cm.cool)
norm = matplotlib.colors.Normalize(min(pos_), max(pos_))
cbar = plt.colorbar(cm.ScalarMappable(norm=norm, 
        cmap=plt.cm.cool))
cbar.set_label('ybinned', rotation=270)
plt.title(f"Rew loc = {rewloc}\n Epoch = {ep}")

# check other latents
# fig = plt.figure(figsize=(10,5))
# ax1 = fig.add_subplot(131, projection="3d")
# ax2 = fig.add_subplot(132, projection="3d")
# ax3 = fig.add_subplot(133, projection="3d")     

# ax1 = cebra.plot_embedding(embedding, embedding_labels=pos_, 
# idx_order=(0,1,2), title="Latents: (1,2,3)", ax=ax1)
# ax2 = cebra.plot_embedding(embedding, embedding_labels=pos_, 
# idx_order=(3,4,5), title="Latents: (4,5,6)", ax=ax2)
# ax3 = cebra.plot_embedding(embedding, embedding_labels=pos_, 
# idx_order=(5,6,7), title="Latents: (6,7,8)", ax=ax3)

cebra.plot_loss(single_cebra_model_pos)
cebra.plot_temperature(single_cebra_model_pos)