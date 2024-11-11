
"""
zahra
july 2024
quantify reward-relative cells
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from placecell import make_tuning_curves,make_tuning_curves_radians_by_trialtype, intersect_arrays
from rewardcell import get_radian_position,create_mask_from_coordinates,pairwise_distances
from projects.opto.behavior.behavior import get_success_failure_trials
from projects.memory.dopamine import get_rewzones
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
savepth = os.path.join(savedst, 'reward_and_pcs_with_masks.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_reward_cell_bytrialtype_nopto_20241108.p"
with open(saveddataset, "rb") as fp: #unpickle
        radian_alignment_saved = pickle.load(fp)
#%%
# initialize var
# radian_alignment_saved = {} # overwrite
goal_cell_iind = []
goal_cell_prop = []
goal_cell_null = []
dist_to_rew = [] # per epoch
num_epochs = []
rates_all = []
total_cells = []
epoch_perm = []
radian_alignment = {}
av_pairwise_dist = []
pcs_per_2ep=[]
cm_window = 20
celldf = []
# cm_window = [10,20,30,40,50,60,70,80] # cm
# iterate through all animals

for ii in range(len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    if (animal!='e217') & (conddf.optoep.values[ii]==-1):
        if animal=='e145': pln=2 
        else: pln=0
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
        print(params_pth)
        fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
            'pyr_tc_s2p_cellind', 'ybinned', 'VR', 'forwardvel', 'trialnum', 
            'rewards', 'iscell', 'bordercells',
            'putative_pcs','stat'])
        VR = fall['VR'][0][0][()]
        scalingf = VR['scalingFACTOR'][0][0]
        try:
                rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
        except:
                rewsize = 10
        ybinned = fall['ybinned'][0]/scalingf
        track_length=180/scalingf    
        forwardvel = fall['forwardvel'][0]    
        changeRewLoc = np.hstack(fall['changeRewLoc'])
        trialnum=fall['trialnum'][0]
        rewards = fall['rewards'][0]
        if animal=='e145':
                ybinned=ybinned[:-1]
                forwardvel=forwardvel[:-1]
                changeRewLoc=changeRewLoc[:-1]
                trialnum=trialnum[:-1]
                rewards=rewards[:-1]
        # set vars
        eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
        lasttr=8 # last trials
        bins=90
        rad = get_radian_position(eps,ybinned,rewlocs,track_length,rewsize) # get radian coordinates
        track_length_rad = track_length*(2*np.pi/track_length)
        bin_size=track_length_rad/bins        
        # get average success rate
        rates = []
        for ep in range(len(eps)-1):
                eprng = range(eps[ep],eps[ep+1])
                success, fail, str_trials, ftr_trials, ttr, \
                total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
                rates.append(success/total_trials)
        rates_all.append(np.array(rates))
        # added to get anatomical info
        # takes time
        fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
        Fc3 = fall_fc3['Fc3']
        dFF = fall_fc3['dFF']
        Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
        dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
        skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
        # skew_filter = skew[((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
        # skew_mask = skew_filter>2
        Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
        if f'{animal}_{day:03d}_index{ii:03d}' in radian_alignment_saved.keys():
            tcs_correct, coms_correct, tcs_fail, coms_fail, \
            com_goal, goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av = radian_alignment_saved[f'{animal}_{day:03d}_index{ii:03d}']            
        else:# remake tuning curves relative to reward        
            # 9/19/24
            # find correct trials within each epoch!!!!
            tcs_correct, coms_correct, tcs_fail, coms_fail = make_tuning_curves_radians_by_trialtype(eps,rewlocs,ybinned,rad,Fc3,trialnum,
                rewards,forwardvel,rewsize,bin_size)          
        if len(coms_correct)==3: # only look at 3 epochs
            fall_stat = scipy.io.loadmat(params_pth, variable_names=['stat','ops'])
            ops = fall_stat['ops']
            stat = fall_stat['stat']
            meanimg=np.squeeze(ops)[()]['meanImg']
            s2p_iind = np.arange(stat.shape[1])
            s2p_iind_filter = s2p_iind[((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
            s2p_iind_filter = s2p_iind_filter[skew>2]
            goal_window = cm_window*(2*np.pi/track_length) # cm converted to rad
            # change to relative value 
            coms_rewrel = np.array([com-np.pi for com in coms_correct])
            perm = list(combinations(range(len(coms_correct)), 2))     
            # if 4 ep
            com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
            com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
            dist_to_rew.append(coms_rewrel)
            # get goal cells across all epochs        
            goal_cells = intersect_arrays(*com_goal) 
            s2p_iind_goal_cells = s2p_iind_filter[goal_cells]
            # find those that stop being goal cells
            dropped_cells = [[xx for xx in comg \
                if xx not in goal_cells] for comg in com_goal]
            # proportion of drop per epoch 
            # dropped_cells_p = [len(xx)/len(com_goal[ii]) for ii,xx in enumerate(dropped_cells)]
            # lets say this is around 80%
            # what do they do in the other epochs?
            # tuning
            rz = get_rewzones(rewlocs,gainf=1/scalingf)
            dfs=[]
            com_droppped_cells = [coms_correct[2,xx] for xx in dropped_cells[0]]
            com_1ep_droppped_cells = [coms_correct[0,xx] for xx in dropped_cells[0]]
            com_2ep_droppped_cells = [coms_correct[1,xx] for xx in dropped_cells[0]]
        # plt.figure()
        # plt.scatter(com_1ep_droppped_cells,com_2ep_droppped_cells, label = 'Comparison Epoch',color='k')
        # plt.scatter(com_1ep_droppped_cells,com_droppped_cells, label='Ref. Epoch 0')
        # plt.scatter(com_2ep_droppped_cells,com_droppped_cells,label='Ref. Epoch 1')
        # plt.title('ep0vs1, ep2')
        # plt.xlabel('Ref Epoch')
        # plt.ylabel('Dropped Epoch')
        # plt.legend()
        # com_droppped_cells = [coms_correct[1,xx] for xx in dropped_cells[1]]
        # com_1ep_droppped_cells = [coms_correct[0,xx] for xx in dropped_cells[1]]
        # com_2ep_droppped_cells = [coms_correct[2,xx] for xx in dropped_cells[1]]
        # plt.figure()
        # plt.scatter(com_1ep_droppped_cells,com_2ep_droppped_cells, label = 'Comparison Epoch',color='k')
        # plt.scatter(com_1ep_droppped_cells,com_droppped_cells, label='Ref. Epoch 0')
        # plt.scatter(com_2ep_droppped_cells,com_droppped_cells,label='Ref. Epoch 2')
        # plt.title('ep0vs2, ep1')
        # plt.xlabel('Ref Epoch')
        # plt.ylabel('Dropped Epoch')
        # plt.legend()
        # com_droppped_cells = [coms_correct[0,xx] for xx in dropped_cells[2]]
        # com_1ep_droppped_cells = [coms_correct[1,xx] for xx in dropped_cells[2]]
        # com_2ep_droppped_cells = [coms_correct[2,xx] for xx in dropped_cells[2]]
        # plt.figure(); plt.scatter(com_1ep_droppped_cells,com_2ep_droppped_cells, label = 'Comparison Epoch',color='k')
        # plt.scatter(com_1ep_droppped_cells,com_droppped_cells, label='Ref. Epoch 1')
        # plt.scatter(com_2ep_droppped_cells,com_droppped_cells,label='Ref. Epoch 2')
        # plt.title('ep1vs2, ep0')
        # plt.xlabel('Ref Epoch')
        # plt.ylabel('Dropped Epoch')
        # plt.legend()

        # for ii in range(len(dropped_cells[0])):
        #     plt.figure()
        #     plt.plot(tcs_1ep_droppped_cells[ii],color='k')
        #     plt.plot(tcs_2ep_droppped_cells[ii],color='slategray')
        #     plt.plot(tcs_droppped_cells[ii],color='darkcyan')
            # plt.legend()
            fc3_droppped_cells = [np.nanmean(tcs_correct[2,xx]) for xx in dropped_cells[0]]
            fc3_1ep_droppped_cells = [np.nanmean(tcs_correct[0,xx]) for xx in dropped_cells[0]]
            fc3_2ep_droppped_cells = [np.nanmean(tcs_correct[1,xx]) for xx in dropped_cells[0]]
            df=pd.DataFrame()
            df['activity'] = np.concatenate([fc3_droppped_cells,fc3_1ep_droppped_cells,fc3_2ep_droppped_cells])
            df['com'] = np.concatenate([com_droppped_cells,com_1ep_droppped_cells,
                            com_2ep_droppped_cells])
            df['cellid'] = np.concatenate([dropped_cells[0]]*3)
            df['epoch'] = np.concatenate([['dropped_ep_ep2_0v1']*len(fc3_droppped_cells),
                                    ['comp_ep0']*len(fc3_1ep_droppped_cells),
                                    ['comp_ep1']*len(fc3_2ep_droppped_cells)])
            # add rewzone
            df['rewzone'] = np.concatenate([[rz[2]]*len(fc3_droppped_cells),
                        [rz[0]]*len(fc3_1ep_droppped_cells),
                        [rz[1]]*len(fc3_2ep_droppped_cells)])
            dfs.append(df)# g=sns.barplot(x='epoch', y='activity',hue='epoch',data=df,fill=False,errorbar='se')
            # sns.stripplot(x='epoch', y='activity',hue='epoch',data=df,s=8)
            # for ii in range(len(dropped_cells[0])):
            #     df_cll = df[df.cellid==ii]
            #     sns.lineplot(x='epoch', y='activity',data=df_cll,errorbar=None,
            #         color='dimgrey',alpha=0.5)
            # g.set_ylim(0,.2)
        # g.set_yscale("log",base=2)

        
            com_droppped_cells = [coms_correct[1,xx] for xx in dropped_cells[1]]
            com_1ep_droppped_cells = [coms_correct[0,xx] for xx in dropped_cells[1]]
            com_2ep_droppped_cells = [coms_correct[2,xx] for xx in dropped_cells[1]]        
            # plt.scatter(com_1ep_droppped_cells,com_2ep_droppped_cells, label = 'goal_cells_within_ep',color='k')
            # plt.scatter(com_1ep_droppped_cells,com_droppped_cells, label='goal_cells_outside_ep2v1')
            # plt.scatter(com_2ep_droppped_cells,com_droppped_cells,label='goal_cells_outside_ep3v1')
            fc3_droppped_cells = [np.nanmean(tcs_correct[1,xx]) for xx in dropped_cells[1]]
            fc3_1ep_droppped_cells = [np.nanmean(tcs_correct[0,xx]) for xx in dropped_cells[1]]
            fc3_2ep_droppped_cells = [np.nanmean(tcs_correct[2,xx]) for xx in dropped_cells[1]]
            df=pd.DataFrame()
            df['activity'] = np.concatenate([fc3_droppped_cells,fc3_1ep_droppped_cells,fc3_2ep_droppped_cells])
            df['com'] = np.concatenate([com_droppped_cells,com_1ep_droppped_cells,
                            com_2ep_droppped_cells])            
            df['cellid'] = np.concatenate([dropped_cells[1]]*3)
            df['epoch'] = np.concatenate([['dropped_ep1_0v2']*len(fc3_droppped_cells),
                                    ['comp_ep0']*len(fc3_1ep_droppped_cells),
                                    ['comp_ep2']*len(fc3_2ep_droppped_cells)])
            # add rewzone
            df['rewzone'] = np.concatenate([[rz[1]]*len(fc3_droppped_cells),
                        [rz[0]]*len(fc3_1ep_droppped_cells),
                        [rz[2]]*len(fc3_2ep_droppped_cells)])

            dfs.append(df)
            com_droppped_cells = [coms_correct[0,xx] for xx in dropped_cells[2]]
            com_1ep_droppped_cells = [coms_correct[1,xx] for xx in dropped_cells[2]]
            com_2ep_droppped_cells = [coms_correct[2,xx] for xx in dropped_cells[2]]        
            fc3_droppped_cells = [np.nanmean(tcs_correct[0,xx]) for xx in dropped_cells[2]]
            fc3_1ep_droppped_cells = [np.nanmean(tcs_correct[1,xx]) for xx in dropped_cells[2]]
            fc3_2ep_droppped_cells = [np.nanmean(tcs_correct[2,xx]) for xx in dropped_cells[2]]
            df=pd.DataFrame()
            df['activity'] = np.concatenate([fc3_droppped_cells,fc3_1ep_droppped_cells,fc3_2ep_droppped_cells])
            df['com'] = np.concatenate([com_droppped_cells,com_1ep_droppped_cells,
                            com_2ep_droppped_cells])            
            df['cellid'] = np.concatenate([dropped_cells[2]]*3)
            df['epoch'] = np.concatenate([['dropped_ep0_1v2']*len(fc3_droppped_cells),
                                    ['comp_ep1']*len(fc3_1ep_droppped_cells),
                                    ['comp_ep2']*len(fc3_2ep_droppped_cells)])
            # add rewzone
            df['rewzone'] = np.concatenate([[rz[0]]*len(fc3_droppped_cells),
                        [rz[1]]*len(fc3_1ep_droppped_cells),
                        [rz[2]]*len(fc3_2ep_droppped_cells)])

            dfs.append(df)
            bigdf = pd.concat(dfs)
            bigdf['animal'] = [animal]*len(bigdf)
            bigdf['day'] = [day]*len(bigdf)
            celldf.append(bigdf)
        # plt.legend()
        # plt.legend()

        # did a check to see if max com misses any of them, they dont seem v compelling
        # bin_size=track_length_rad/bins            
        # max_com = [np.arange(0,track_length_rad,bin_size)[np.argmax(tcs_correct[i,dropped_cells,:],
        #                 axis=1)] for i in range(len(coms_correct))]
        
            # get per comparison
            goal_cells_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in com_goal]
            goal_cell_iind.append(goal_cells);goal_cell_p=len(goal_cells)/len(coms_correct[0])
            epoch_perm.append(perm)
            goal_cell_prop.append([goal_cells_p_per_comparison,goal_cell_p]);num_epochs.append(len(coms_correct))

            #if pc in one epoch
            pcs = np.vstack(np.array(fall['putative_pcs'][0]))
            pc_bool = np.sum(pcs,axis=0)>=1      
            Fc3 = fall_fc3['Fc3']
            Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
            Fc3 = Fc3[:,((skew>2)&pc_bool)] # only keep cells with skew greateer than 2
            if Fc3.shape[1]==0: # remove skew filter if v few or noisy cells
                Fc3 = fall_fc3['Fc3']
                Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
                Fc3 = Fc3[:,(pc_bool)] 
                s2p_iind_filter = s2p_iind[((fall['iscell'][:,0]).astype(bool) \
                    & (~fall['bordercells'][0].astype(bool)))]
            bin_size=3 # cm
            # get abs dist tuning 
            tcs_correct_abs, coms_correct_abs = make_tuning_curves(eps,rewlocs,ybinned,
                Fc3,trialnum,rewards,forwardvel,
                rewsize,bin_size)

            # get cells that maintain their coms across at least 2 epochs
            place_window = 10 # cm converted to rad                
            perm = list(combinations(range(len(coms_correct_abs)), 2))     
            com_per_ep = np.array([(coms_correct_abs[perm[jj][0]]-coms_correct_abs[perm[jj][1]]) for jj in range(len(perm))])        
            compc = [np.where((comr<place_window) & (comr>-place_window))[0] for comr in com_per_ep]
            # get place cells that are place-like for at least two epochs
            pcall = np.concatenate(compc)
            # get % per comparison
            pcs_p_per_comparison = [len(xx)/len(coms_correct_abs[0]) for xx in compc]
            pcs_per_2ep.append(pcs_p_per_comparison)
            s2p_iind_pc = s2p_iind_filter[pcall]
            # s2p_iind_goal_cells_that_are_pc = s2p_iind_filter[goal_cells_that_are_pc]
            # Create a colormap
            cmap = ['k','yellow']  # Choose your preferred colormap
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", cmap)
            cmap.set_under('none')
            cmap2 = ['k','cyan']  # Choose your preferred colormap
            cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", cmap2)
            cmap2.set_under('none')
            # get x y coords!!
            fig,ax = plt.subplots()
            stat = np.squeeze(stat)        
            ax.imshow(meanimg, cmap='Greys_r')
            centerspc = []; centersgc = []        
            for gc in s2p_iind_pc:
                    ypix=stat[gc][0][0][0][0]    
                    xpix=stat[gc][0][0][1][0]     
                    coords = np.column_stack((xpix, ypix))  
                    mask,cmask,center=create_mask_from_coordinates(coords, 
                            meanimg.shape)                         
                    ax.imshow(cmask,cmap=cmap2,vmin=1)
                    centerspc.append(center)          
            for gc in s2p_iind_goal_cells:
                    ypix=stat[gc][0][0][0][0]    
                    xpix=stat[gc][0][0][1][0]     
                    coords = np.column_stack((xpix, ypix))  
                    mask,cmask,center=create_mask_from_coordinates(coords, 
                            meanimg.shape)                         
                    ax.imshow(cmask,cmap=cmap,vmin=1)
                    centersgc.append(center)   
            ax.axis('off')
            fig.suptitle(f'animal: {animal}, day: {day}')
            pdf.savefig(fig)
            plt.close(fig)
            points = np.array(centersgc)
            dist = pairwise_distances(points)
            # Exclude self-pairs (distances to the same point)
            # We do this by flattening the distance matrix and excluding zero distances
            non_self_distances = dist[np.triu_indices_from(dist, k=1)]
            # Compute the average distance
            average_distance_gc = np.mean(non_self_distances)
            points = np.array(centerspc)
            # pcs
            dist = pairwise_distances(points)        
            non_self_distances = dist[np.triu_indices_from(dist, k=1)]
            average_distance_pc = np.mean(non_self_distances)
            av_pairwise_dist.append([average_distance_gc,average_distance_pc])
            colors = ['k', 'slategray', 'darkcyan', 'darkgoldenrod', 'orchid']
            if len(goal_cells)>0:
                rows = int(np.ceil(np.sqrt(len(goal_cells))))
                cols = int(np.ceil(len(goal_cells) / rows))
                fig, axes = plt.subplots(rows, cols, figsize=(15, 15),sharex=True)
                if len(goal_cells) > 1:
                    axes = axes.flatten()
                else:
                    axes = [axes]
                for i,gc in enumerate(goal_cells):            
                    for ep in range(len(coms_correct)):
                        ax = axes[i]
                        ax.plot(tcs_correct[ep,gc,:], label=f'rewloc {rewlocs[ep]}', color=colors[ep])
                        if len(tcs_fail)>0:
                                ax.plot(tcs_fail[ep,gc,:], label=f'fail rewloc {rewlocs[ep]}', color=colors[ep], linestyle = '--')
                        ax.axvline((bins/2), color='k')
                        ax.set_title(f'cell # {gc}')
                        ax.spines[['top','right']].set_visible(False)
                ax.set_xticks(np.arange(0,bins+1,20))
                ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi, np.pi/2.25),3))
                ax.set_xlabel('Radian position (centered start rew loc)')
                ax.set_ylabel('Fc3')
                ax.legend()
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)


            # place cells
            rows = int(np.ceil(np.sqrt(len(pcall))))
            cols = int(np.ceil(len(pcall) / rows))
            fig, axes = plt.subplots(rows, cols, figsize=(30, 30),sharex=True)
            if len(pcall) > 1:
                axes = axes.flatten()
            else:
                axes = [axes]
            for i,gc in enumerate(pcall):            
                for ep in range(len(coms_correct_abs)):
                    ax = axes[i]
                    ax.plot(tcs_correct_abs[ep,gc,:], label=f'rewloc {rewlocs[ep]}', color=colors[ep])
                    ax.set_title(f'place cell # {gc}')
                    ax.spines[['top','right']].set_visible(False)
            ax.set_xticks(np.arange(0,(track_length/bin_size)+bin_size,10))
            ax.set_xticklabels(np.arange(0,track_length+bin_size*10,bin_size*10).astype(int))
            ax.set_xlabel('Absolute position (cm)')
            ax.set_ylabel('Fc3')
            ax.legend()
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            total_cells.append(len(coms_correct[0]))
            radian_alignment[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_correct, coms_correct, tcs_fail, coms_fail,
                            com_goal,compc]

pdf.close()
# save pickle of dcts
# with open(saveddataset, "wb") as fp:   #Pickling
#     pickle.dump(radian_alignment, fp) 
#%%
# pairwise distances goal vs. place
df = pd.DataFrame()
plt.rc('font', size=22)
df['mean_pairwise_distance_px'] = np.concatenate(av_pairwise_dist)
df['cell_type'] = np.concatenate(np.column_stack(np.array([['goal_cell']*int(len(df)/2),['place_cell']*int(len(df)/2)])))
fig,ax = plt.subplots(figsize=(2.2,5))
# av across mice
sns.stripplot(x='cell_type', y='mean_pairwise_distance_px',hue='cell_type',
        data=df,
        s=10,alpha=0.5)
sns.barplot(x='cell_type', y='mean_pairwise_distance_px',hue='cell_type',
        data=df, fill=False,ax=ax, errorbar='se')
ax.spines[['top','right']].set_visible(False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
# fig.tight_layout()
# %%
plt.rc('font', size=30) 
# cell df dropped vs. goal cell epochs
df = pd.concat(celldf)
# df = df[df.animal=='e201']
df = df.reset_index()
fig,ax = plt.subplots(figsize=(20,10))
orderlst=['dropped_ep_ep2_0v1','dropped_ep1_0v2',
       'dropped_ep0_1v2','comp_ep0','comp_ep1','comp_ep2']
g=sns.barplot(x='epoch', y='activity',hue='animal',
        data=df,fill=False,errorbar='se',order=orderlst)
# sns.stripplot(x='epoch', y='activity',hue='epoch',data=df,s=8)

for ii in range(len(dropped_cells[0])):
    df_cll = df[df.cellid==ii]
    sns.lineplot(x='epoch', y='activity',
        data=df_cll,errorbar=None,
        color='dimgrey',alpha=0.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
g.set_yscale("log")
ax.legend(bbox_to_anchor=(1.1, 1.05))

#%%
# tuning properties of dropped cells

plt.rc('font', size=30) 
# cell df dropped vs. goal cell epochs
df = pd.concat(celldf)
df = df.reset_index()
an='e190'
dfan = df[df.animal==an]

fig,ax = plt.subplots(figsize=(20,10))
orderlst=['dropped_ep_ep2_0v1','dropped_ep1_0v2',
       'dropped_ep0_1v2','comp_ep0','comp_ep1','comp_ep2']
g=sns.stripplot(x='epoch', y='com',hue='day',
        data=dfan,order=orderlst)
# sns.stripplot(x='epoch', y='activity',hue='epoch',data=df,s=8)

# for an in df.animal.unique():
for dy in dfan.day.unique():
    dfandy = dfan[dfan.day==dy]
    for ii in dfandy.cellid.unique():
        df_cll = dfandy[dfandy.cellid==ii]
        sns.lineplot(x='epoch', y='com',
            data=df_cll,errorbar=None,
            color='dimgrey',alpha=0.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    # g.set_yscale("log")
    ax.legend(bbox_to_anchor=(1.1, 1.05))
ax.axhline(np.pi, linestyle='--',color='k')
fig.suptitle(f'animal: {an}')
