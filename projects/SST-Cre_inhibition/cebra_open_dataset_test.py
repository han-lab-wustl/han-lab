import scipy.io
import numpy as np, os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
matplotlib.use('TkAgg') #might need for gui
import cebra
from cebra import CEBRA

mat = r'Y:\sstcre_analysis\hrz\cebra\open_datasets\tests_hc30.mat'
dst = r'Y:\sstcre_analysis\hrz\cebra\saved_models'

neural_data = cebra.load_data(file=mat, key='traces57')
data = neural_data
ybinned = cebra.load_data(file=mat, 
            key="position57")

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
single_cebra_model_pos.fit(data, ybinned)
embedding = single_cebra_model_pos.transform(data)
# discrete var
cebra.plot_embedding(embedding, np.squeeze(ybinned), cmap = plt.cm.cool, markersize=5)
norm = matplotlib.colors.Normalize(min(ybinned), max(ybinned))
cbar = plt.colorbar(cm.ScalarMappable(norm=norm, 
        cmap=plt.cm.cool))
cbar.set_label('ybinned', rotation=270)
# plt.title(f"Rew loc = {rewloc}\n Epoch = {ep}")
plt.ion()

single_cebra_model_pos.save(os.path.join(dst, f'dim3_traces57_schnitzerlab_miniscope.pt'))

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