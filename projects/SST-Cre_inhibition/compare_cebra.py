# compare cebra across animals
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

srcdir = r'Y:\sstcre_analysis\hrz\cebra\saved_models\multisession\control'
models = ['dim3_e186_days1-8_ep2_last8trials_successtrials_before_rewloc.pt', 
          'dim3_e200_ctrluntilday89_ep2_last8trials_successtrials_before_rewloc.pt',
          'dim3_e201_ctrluntilday85_ep2_last8trials_successtrials_before_rewloc.pt',
          'dim3_e145_day4-12_rewzone3_last8trials_successtrials_before_rewloc.pt']
animals = ['e186', 'e200', 'e201', 'e145']
        
dataz = []; posz = []
embeddings = [];
for j,animal in enumerate(animals):
    if animal == 'e186':
        days = np.arange(1,9)#np.arange(1,9)#[67,70,73,76,80,83,86,89]
        src = r'Z:\cellreg1month_Fmats\E186'#r'Y:\sstcre_imaging\e200'
        mat = []
        for day in days:       
            #    mat.append(glob.glob(os.path.join(src, f'{day}', '**', '*Fall.mat'), recursive = True)[0])
            mat.append(glob.glob(os.path.join(src, f'd{day:03d}', '**', '*Fall.mat'), recursive = True)[0])
    elif animal == 'e200':
        days = [67,70,73,76,80,83,86,89]
        src = r'Y:\sstcre_imaging\e200'
        mat = []
        for day in days:
            mat.append(glob.glob(os.path.join(src, f'{day}', '**', '*Fall.mat'), recursive = True)[0])
    elif animal == 'e201':
        days = [57, 60, 63, 66, 69, 72, 76, 79, 85]
        src = r'Z:\sstcre_imaging\e201'
        mat = []
        for day in days:
            mat.append(glob.glob(os.path.join(src, f'{day}', '**', '*Fall.mat'), recursive = True)[0])
    elif animal == 'e145':        
        src = r'X:\pyramidal_cell_data\e145'
        mat = [os.path.join(src, xx) for xx in os.listdir(src)]
        

    i = 0 # only test for session 4 in each multisession embedding

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
        
    if data.shape[0]>0:
        dataz.append(data)
        posz.append(pos_)
    else:
        print(f'\n day {i+1} not used, not enuf successful trials')

    # load model
    cebra_model = cebra.CEBRA.load(os.path.join(srcdir, models[j]))
    embeddings.append(cebra_model.transform(data, session_id=i))#, pos_)) 
       


# Between-datasets, by ali gning on the labels
(scores_datasets,
    pairs_datasets,
    datasets_datasets) = cebra.sklearn.metrics.consistency_score(embeddings=embeddings,
        labels=posz,
        between="datasets")

plt.figure(figsize=(10,4))

cebra.plot_consistency(scores_datasets, pairs_datasets, datasets_datasets, 
    vmin=0, vmax=100, title="Between-subjects consistencies")
plt.xticks(range(len(dataz)), animals)
plt.yticks(range(len(dataz)), animals)

for k,embedding in enumerate(embeddings):
    cebra.plot_embedding(embedding, posz[k], cmap = plt.cm.cool, markersize=5)
    norm = matplotlib.colors.Normalize(min(pos_), max(pos_))
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, 
            cmap=plt.cm.cool))
    cbar.set_label('ybinned', rotation=270)
    plt.title(f"{animals[k]}")
    plt.ion()
