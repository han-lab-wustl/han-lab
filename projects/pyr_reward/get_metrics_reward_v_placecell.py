
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
pvals = []
rates_all = []
total_cells = []
epoch_perm = []
radian_alignment = {}
av_pairwise_dist = []
pcs_per_2ep=[]
cm_window = 20
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
        rates_all.append(np.nanmean(np.array(rates)))
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
        com_droppped_cells = [coms_correct[2,xx] for xx in dropped_cells[0]]
        com_1ep_droppped_cells = [coms_correct[0,xx] for xx in dropped_cells[0]]
        com_2ep_droppped_cells = [coms_correct[1,xx] for xx in dropped_cells[0]]
        plt.figure()
        plt.scatter(com_1ep_droppped_cells,com_2ep_droppped_cells, label = 'Comparison Epoch',color='k')
        plt.scatter(com_1ep_droppped_cells,com_droppped_cells, label='Ref. Epoch 0')
        plt.scatter(com_2ep_droppped_cells,com_droppped_cells,label='Ref. Epoch 1')
        plt.title('ep0vs1, ep2')
        plt.xlabel('Ref Epoch')
        plt.ylabel('Dropped Epoch')
        plt.legend()
        com_droppped_cells = [coms_correct[1,xx] for xx in dropped_cells[1]]
        com_1ep_droppped_cells = [coms_correct[0,xx] for xx in dropped_cells[1]]
        com_2ep_droppped_cells = [coms_correct[2,xx] for xx in dropped_cells[1]]
        plt.figure()
        plt.scatter(com_1ep_droppped_cells,com_2ep_droppped_cells, label = 'Comparison Epoch',color='k')
        plt.scatter(com_1ep_droppped_cells,com_droppped_cells, label='Ref. Epoch 0')
        plt.scatter(com_2ep_droppped_cells,com_droppped_cells,label='Ref. Epoch 2')
        plt.title('ep0vs2, ep1')
        plt.xlabel('Ref Epoch')
        plt.ylabel('Dropped Epoch')
        plt.legend()
        com_droppped_cells = [coms_correct[0,xx] for xx in dropped_cells[2]]
        com_1ep_droppped_cells = [coms_correct[1,xx] for xx in dropped_cells[2]]
        com_2ep_droppped_cells = [coms_correct[2,xx] for xx in dropped_cells[2]]
        plt.figure(); plt.scatter(com_1ep_droppped_cells,com_2ep_droppped_cells, label = 'Comparison Epoch',color='k')
        plt.scatter(com_1ep_droppped_cells,com_droppped_cells, label='Ref. Epoch 1')
        plt.scatter(com_2ep_droppped_cells,com_droppped_cells,label='Ref. Epoch 2')
        plt.title('ep1vs2, ep0')
        plt.xlabel('Ref Epoch')
        plt.ylabel('Dropped Epoch')
        plt.legend()

        # v. activity
        tcs_droppped_cells = [tcs_correct[2,xx] for xx in dropped_cells[0]]
        tcs_1ep_droppped_cells = [tcs_correct[0,xx] for xx in dropped_cells[0]]
        tcs_2ep_droppped_cells = [tcs_correct[1,xx] for xx in dropped_cells[0]]
                
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
        df['cellid'] = np.concatenate([np.arange(len(dropped_cells[0]))]*3)
        df['epoch'] = np.concatenate([['dropped_ep']*len(fc3_droppped_cells),
                                ['comp_ep_1']*len(fc3_1ep_droppped_cells),
                                ['comp_ep_2']*len(fc3_2ep_droppped_cells)])
        g=sns.barplot(x='epoch', y='activity',hue='epoch',data=df,fill=False,errorbar='se')
        sns.stripplot(x='epoch', y='activity',hue='epoch',data=df,s=8)
        for ii in range(len(dropped_cells[0])):
            df_cll = df[df.cellid==ii]
            sns.lineplot(x='epoch', y='activity',data=df_cll,errorbar=None,
                color='dimgrey',alpha=0.5)
        g.set_ylim(0,.2)
        # g.set_yscale("log",base=2)

        
        com_droppped_cells = [coms_correct[1,xx] for xx in dropped_cells[1]]
        com_1ep_droppped_cells = [coms_correct[0,xx] for xx in dropped_cells[1]]
        com_2ep_droppped_cells = [coms_correct[2,xx] for xx in dropped_cells[1]]        
        plt.scatter(com_1ep_droppped_cells,com_2ep_droppped_cells, label = 'goal_cells_within_ep',color='k')
        plt.scatter(com_1ep_droppped_cells,com_droppped_cells, label='goal_cells_outside_ep2v1')
        plt.scatter(com_2ep_droppped_cells,com_droppped_cells,label='goal_cells_outside_ep3v1')
        fc3_droppped_cells = [np.nanmean(tcs_correct[1,xx]) for xx in dropped_cells[0]]
        fc3_1ep_droppped_cells = [np.nanmean(tcs_correct[0,xx]) for xx in dropped_cells[0]]
        fc3_2ep_droppped_cells = [np.nanmean(tcs_correct[2,xx]) for xx in dropped_cells[0]]
        df=pd.DataFrame()
        df['activity'] = np.concatenate([fc3_droppped_cells,fc3_1ep_droppped_cells,fc3_2ep_droppped_cells])
        df['cellid'] = np.concatenate([np.arange(len(dropped_cells[0]))]*3)
        df['epoch'] = np.concatenate([['dropped_ep']*len(fc3_droppped_cells),
                                ['comp_ep_1']*len(fc3_1ep_droppped_cells),
                                ['comp_ep_2']*len(fc3_2ep_droppped_cells)])
        g=sns.barplot(x='epoch', y='activity',hue='epoch',data=df,fill=False,errorbar='se')
        sns.stripplot(x='epoch', y='activity',hue='epoch',data=df,s=8)
        for ii in range(len(dropped_cells[0])):
            df_cll = df[df.cellid==ii]
            sns.lineplot(x='epoch', y='activity',data=df_cll,errorbar=None,
                color='dimgrey',alpha=0.5)
        # plt.legend()

        com_droppped_cells = [coms_correct[0,xx] for xx in dropped_cells[2]]
        com_1ep_droppped_cells = [coms_correct[1,xx] for xx in dropped_cells[2]]
        com_2ep_droppped_cells = [coms_correct[2,xx] for xx in dropped_cells[2]]        
        plt.scatter(com_1ep_droppped_cells,com_2ep_droppped_cells, label = 'goal_cells_within_ep',color='k')
        plt.scatter(com_1ep_droppped_cells,com_droppped_cells, label='goal_cells_outside_ep2v1')
        plt.scatter(com_2ep_droppped_cells,com_droppped_cells,label='goal_cells_outside_ep3v1')
        plt.xlabel('Epoch 1')
        plt.ylabel('Epoch 2')
        fc3_droppped_cells = [np.nanmean(tcs_correct[0,xx]) for xx in dropped_cells[0]]
        fc3_1ep_droppped_cells = [np.nanmean(tcs_correct[1,xx]) for xx in dropped_cells[0]]
        fc3_2ep_droppped_cells = [np.nanmean(tcs_correct[2,xx]) for xx in dropped_cells[0]]
        df=pd.DataFrame()
        df['activity'] = np.concatenate([fc3_droppped_cells,fc3_1ep_droppped_cells,fc3_2ep_droppped_cells])
        df['cellid'] = np.concatenate([np.arange(len(dropped_cells[0]))]*3)
        df['epoch'] = np.concatenate([['dropped_ep']*len(fc3_droppped_cells),
                                ['comp_ep_1']*len(fc3_1ep_droppped_cells),
                                ['comp_ep_2']*len(fc3_2ep_droppped_cells)])
        g=sns.barplot(x='epoch', y='activity',hue='epoch',data=df,fill=False,errorbar='se')
        sns.stripplot(x='epoch', y='activity',hue='epoch',data=df,s=8)
        for ii in range(len(dropped_cells[0])):
            df_cll = df[df.cellid==ii]
            sns.lineplot(x='epoch', y='activity',data=df_cll,errorbar=None,
                color='dimgrey',alpha=0.5)
        g.set_ylim(0,.1)
        # plt.legend()
        # plt.legend()

        # did a check to see if max com misses any of them, they dont seem v compelling
        # bin_size=track_length_rad/bins            
        # max_com = [np.arange(0,track_length_rad,bin_size)[np.argmax(tcs_correct[i,dropped_cells,:],
        #                 axis=1)] for i in range(len(coms_correct))]
        
        # redo with max com
        # change to relative value 
        coms_rewrel = np.array([com-np.pi for com in max_com])
        perm = list(combinations(range(len(coms_correct)), 2))     
        # if 4 ep
        com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
        com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
        s2p_iind_goal_cells_2ep = s2p_iind_filter[goal_cells_per2ep]
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
        s2p_iind_goal_cells_that_are_pc = s2p_iind_filter[goal_cells_that_are_pc]
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
        # goal cells that are goal cells for 2 ep but otherwise not?
        # colors = ['k', 'slategray', 'darkcyan', 'darkgoldenrod', 'orchid']
        # rows = int(np.ceil(np.sqrt(len(goal_cells_per2ep))))
        # cols = int(np.ceil(len(goal_cells_per2ep) / rows))
        # fig, axes = plt.subplots(rows, cols, figsize=(50,50),sharex=True)
        # if len(goal_cells_per2ep) > 1:
        #     axes = axes.flatten()
        # else:
        #     axes = [axes]
        # for i,gc in enumerate(goal_cells_per2ep):            
        #     for ep in range(len(coms_correct)):
        #         ax = axes[i]
        #         ax.plot(tcs_correct[ep,gc,:], label=f'rewloc {rewlocs[ep]}', color=colors[ep])
        #         ax.axvline((bins/2), color='k')
        #         ax.set_title(f'cell # {gc}')
        #         ax.spines[['top','right']].set_visible(False)
        # ax.set_xticks(np.arange(0,bins+1,20))
        # ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi, np.pi/2.25),3))
        # ax.set_xlabel('Radian position (centered start rew loc)')
        # ax.set_ylabel('Fc3')
        # ax.legend()
        # fig.suptitle('Goal cells in at least 2 ep')
        # fig.tight_layout()
        # pdf.savefig(fig)
        # plt.close(fig)

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
        # radian_alignment[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_correct, coms_correct, tcs_fail, coms_fail,
        #                 com_goal, goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av]

pdf.close()
# save pickle of dcts
# with open(saveddataset, "wb") as fp:   #Pickling
#     pickle.dump(radian_alignment, fp) 
#%%
plt.rc('font', size=16)          # controls default text sizes
# plot goal cells across epochs
inds = [int(xx[-3:]) for xx in radian_alignment.keys()]
df = conddf.copy()
df = df[((df.animals!='e217')) & (df.optoep==-1) & (df.index.isin(inds))]
df['num_epochs'] = num_epochs
df['goal_cell_prop'] = [xx[1] for xx in goal_cell_prop]
df['opto'] = df.optoep.values>1
df['condition'] = ['vip' if xx=='vip' else 'ctrl' for xx in df.in_type.values]
df['p_value'] = pvals
df['goal_cell_prop_shuffle'] = [xx[1] for xx in goal_cell_null]

fig,ax = plt.subplots(figsize=(5,5))
ax = sns.histplot(data = df.loc[df.opto==False], x='p_value', 
                hue='animals', bins=40)
ax.spines[['top','right']].set_visible(False)
ax.axvline(x=0.05, color='k', linestyle='--')
sessions_sig = sum(df.loc[df.opto==False,'p_value'].values<0.05)/len(df.loc[df.opto==False])
ax.set_title(f'{(sessions_sig*100):.2f}% of sessions are significant')
ax.set_xlabel('P-value')
ax.set_ylabel('Sessions')
#%%
# number of epochs vs. reward cell prop    
fig,ax = plt.subplots(figsize=(5,5))
df_plt = df[(df.opto==False)]
# av across mice
df_plt = df_plt.groupby(['animals','num_epochs']).mean(numeric_only=True)
sns.stripplot(x='num_epochs', y='goal_cell_prop',
        hue='animals',data=df_plt,
        s=10)
sns.barplot(x='num_epochs', y='goal_cell_prop',
        data=df_plt,
        fill=False,ax=ax, color='k', errorbar='se')
ax = sns.lineplot(data=df_plt, # correct shift
        x=df_plt.index.get_level_values('num_epochs')-2, y='goal_cell_prop_shuffle',color='grey', 
        label='shuffle')
ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05))

eps = [2,3,4]
for ep in eps:
    # rewprop = df_plt.loc[(df_plt.num_epochs==ep), 'goal_cell_prop']
    rewprop = df_plt.loc[(df_plt.index.get_level_values('num_epochs')==ep), 'goal_cell_prop']
    shufprop = df_plt.loc[(df_plt.index.get_level_values('num_epochs')==ep), 'goal_cell_prop_shuffle']
    t,pval = scipy.stats.ranksums(rewprop, shufprop)
    print(f'{ep} epochs, pval: {pval}')
#%%    
# include all comparisons 
df_perms = pd.DataFrame()
df_perms['epoch_comparison'] = [str(tuple(xx)) for xx in np.concatenate(epoch_perm)]
goal_cell_perm = [xx[0] for xx in goal_cell_prop]
goal_cell_perm_shuf = [xx[0][~np.isnan(xx[0])] for xx in goal_cell_null]
df_perms['goal_cell_prop'] = np.concatenate(goal_cell_perm)
df_perms['goal_cell_prop_shuffle'] = np.concatenate(goal_cell_perm_shuf)
df_perm_animals = [[xx]*len(goal_cell_perm[ii]) for ii,xx in enumerate(df.animals.values)]
df_perms['animals'] = np.concatenate(df_perm_animals)
df_perms = df_perms[df_perms.animals!='e189']
df_permsav = df_perms.groupby(['animals','epoch_comparison']).mean(numeric_only=True)

fig,ax = plt.subplots(figsize=(7,5))
sns.stripplot(x='epoch_comparison', y='goal_cell_prop',
        hue='animals',data=df_permsav,
        s=8,ax=ax)
sns.barplot(x='epoch_comparison', y='goal_cell_prop',
        data=df_permsav,
        fill=False,ax=ax, color='k', errorbar='se')
ax = sns.lineplot(data=df_permsav, # correct shift
        x='epoch_comparison', y='goal_cell_prop_shuffle',
        color='grey', label='shuffle')

ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05))

eps = df_permsav.index.get_level_values("epoch_comparison").unique()
for ep in eps:
    # rewprop = df_plt.loc[(df_plt.num_epochs==ep), 'goal_cell_prop']
    rewprop = df_permsav.loc[(df_permsav.index.get_level_values('epoch_comparison')==ep), 'goal_cell_prop'].values
    shufprop = df_permsav.loc[(df_permsav.index.get_level_values('epoch_comparison')==ep), 'goal_cell_prop_shuffle'].values
    t,pval = scipy.stats.ranksums(rewprop, shufprop)
    print(f'{ep} epochs, pval: {pval}')

# take a mean of all epoch comparisons
df_perms['num_epochs'] = [2]*len(df_perms)
df_permsav2 = df_perms.groupby(['animals', 'num_epochs']).mean(numeric_only=True)
#%%
df_plt2 = pd.concat([df_permsav2,df_plt])
# df_plt2 = df_plt2[df_plt2.index.get_level_values('animals')!='e189']
df_plt2 = df_plt2[df_plt2.index.get_level_values('num_epochs')<5]
df_plt2 = df_plt2.groupby(['animals', 'num_epochs']).mean(numeric_only=True)
# number of epochs vs. reward cell prop incl combinations    
fig,ax = plt.subplots(figsize=(3,5))
# av across mice
sns.stripplot(x='num_epochs', y='goal_cell_prop',color='k',
        data=df_plt2,
        s=8)
sns.barplot(x='num_epochs', y='goal_cell_prop',
        data=df_plt2,
        fill=False,ax=ax, color='k', errorbar='se')
ax = sns.lineplot(data=df_plt2, # correct shift
        x=df_plt2.index.get_level_values('num_epochs').astype(int)-2, y='goal_cell_prop_shuffle',color='grey', 
        label='shuffle')
ax.spines[['top','right']].set_visible(False)
ax.legend().set_visible(False)
ax.set_xlabel('# of reward loc. switches')
ax.set_ylabel('Reward cell proportion')
eps = [2,3,4]
y = 0.25
pshift = 0.03
fs=36
for ii,ep in enumerate(eps):
        rewprop = df_plt2.loc[(df_plt2.index.get_level_values('num_epochs')==ep), 'goal_cell_prop']
        shufprop = df_plt2.loc[(df_plt2.index.get_level_values('num_epochs')==ep), 'goal_cell_prop_shuffle']
        t,pval = scipy.stats.ttest_rel(rewprop, shufprop)
        print(f'{ep} epochs, pval: {pval}')
        # statistical annotation        
        if pval < 0.001:
                plt.text(ii, y, "***", ha='center', fontsize=fs)
        elif pval < 0.01:
                plt.text(ii, y, "**", ha='center', fontsize=fs)
        elif pval < 0.05:
                plt.text(ii, y, "*", ha='center', fontsize=fs)
        ax.text(ii-0.5, y+pshift, f'p={pval:.3g}',fontsize=10)

# plt.savefig(os.path.join(savedst, 'reward_cell_prop_per_an.png'), 
#         bbox_inches='tight')
#%%
# pairwise distances goal vs. place
df = pd.DataFrame()

df['mean_pairwise_distance_px'] = np.concatenate(av_pairwise_dist)
df['cell_type'] = np.concatenate(np.column_stack(np.array([['goal_cell']*int(len(df)/2),['place_cell']*int(len(df)/2)])))
fig,ax = plt.subplots(figsize=(3,5))
# av across mice
sns.stripplot(x='cell_type', y='mean_pairwise_distance_px',
        data=df,
        s=8,alpha=0.5)

sns.barplot(x='cell_type', y='mean_pairwise_distance_px',
        data=df, fill=False,ax=ax, errorbar='se')
ax.spines[['top','right']].set_visible(False)

# %%

#%%
# df['recorded_neurons_per_session'] = total_cells
# df_plt_ = df[(df.opto==False)&(df.p_value<0.05)]
# df_plt_= df_plt_[(df_plt_.animals!='e200')&(df_plt_.animals!='e189')]
# df_plt_ = df_plt_.groupby(['animals']).mean(numeric_only=True)

# fig,ax = plt.subplots(figsize=(7,5))
# sns.scatterplot(x='recorded_neurons_per_session', y='goal_cell_prop',hue='animals',
#         data=df_plt_,
#         s=150, ax=ax)
# sns.regplot(x='recorded_neurons_per_session', y='goal_cell_prop',
#         data=df_plt_,
#         ax=ax, scatter=False, color='k'
# )
# r, p = scipy.stats.pearsonr(df_plt_['recorded_neurons_per_session'], 
#         df_plt_['goal_cell_prop'])
# ax = plt.gca()
# ax.text(.5, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
#         transform=ax.transAxes)

# ax.spines[['top','right']].set_visible(False)
# ax.legend(bbox_to_anchor=(1.01, 1.05))
# ax.set_xlabel('Av. # of neurons per session')
# ax.set_ylabel('Reward cell proportion')

# plt.savefig(os.path.join(savedst, 'rec_cell_rew_prop_per_an.svg'), 
#         bbox_inches='tight')

#%%
# # #examples
fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
Fc3 = fall_fc3['Fc3']
dFF = fall_fc3['dFF']
Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
Fc3 = Fc3[:,(skew>2)] # only keep cells with skew greateer than 2
bin_size=3 # cm
# get abs dist tuning 
tcs_correct_abs, coms_correct_abs, tcs_fail, coms_fail = make_tuning_curves_by_trialtype(eps,rewlocs,ybinned,
Fc3,trialnum,rewards,forwardvel,rewsize,bin_size)

# # #plot example tuning curve
plt.rc('font', size=30)  
fig,axes = plt.subplots(1,3,figsize=(20,20), sharex = True)
for ep in range(3):
        axes[ep].imshow(tcs_correct_abs[ep,com_goal[0]][np.argsort(coms_correct_abs[0,com_goal[0]])[:60],:]**.3)
        axes[ep].set_title(f'Epoch {ep+1}')
        axes[ep].axvline((rewlocs[ep]-rewsize/2)/bin_size, color='w', linestyle='--', linewidth=4)
        axes[ep].set_xticks(np.arange(0,(track_length/bin_size)+bin_size,30))
        axes[ep].set_xticklabels(np.arange(0,track_length+bin_size*30,bin_size*30).astype(int))
axes[0].set_ylabel('Reward-distance cells')
axes[2].set_xlabel('Absolute distance (cm)')
plt.savefig(os.path.join(savedst, 'abs_dist_tuning_curves_3_ep.svg'), bbox_inches='tight')

# fig,axes = plt.subplots(1,4,figsize=(15,20), sharey=True, sharex = True)
# axes[0].imshow(tcs_correct[0,com_goal[0]][np.argsort(coms_correct[0,com_goal[0]])[:60],:]**.5)
# axes[0].set_title('Epoch 1')
# im = axes[1].imshow(tcs_correct[1,com_goal[0]][np.argsort(coms_correct[0,com_goal[0]])[:60],:]**.5)
# axes[1].set_title('Epoch 2')
# im = axes[2].imshow(tcs_correct[2,com_goal[0]][np.argsort(coms_correct[0,com_goal[0]])[:60],:]**.5)
# axes[2].set_title('Epoch 3')
# im = axes[3].imshow(tcs_correct[3,com_goal[0]][np.argsort(coms_correct[0,com_goal[0]])[:60],:]**.5)
# axes[3].set_title('Epoch 4')
# ax = axes[1]
# ax.set_xticks(np.arange(0,bins+1,10))
# ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi+np.pi/4.5, np.pi/4.5),1))
# ax.axvline((bins/2), color='w', linestyle='--')
# axes[0].axvline((bins/2), color='w', linestyle='--')
# axes[2].axvline((bins/2), color='w', linestyle='--')
# axes[3].axvline((bins/2), color='w', linestyle='--')
# axes[0].set_ylabel('Reward distance cells')
# axes[3].set_xlabel('Reward-relative distance (rad)')
# fig.tight_layout()
# plt.savefig(os.path.join(savedst, 'tuning_curves_4_ep.png'), bbox_inches='tight')
# for gc in goal_cells:
#%%
gc = 51
plt.rc('font', size=24)  
fig2,ax2 = plt.subplots(figsize=(5,5))

for ep in range(3):        
        ax2.plot(tcs_correct_abs[ep,gc,:], label=f'rewloc {rewlocs[ep]}', color=colors[ep],linewidth=3)
        ax2.axvline(rewlocs[ep]/bin_size, color=colors[ep], linestyle='--',linewidth=3)
        
        ax2.spines[['top','right']].set_visible(False)
ax2.set_title(f'animal: {animal}, day: {day}\ncell # {gc}')
ax2.set_xticks(np.arange(0,(track_length/bin_size)+bin_size,30))
ax2.set_xticklabels(np.arange(0,track_length+bin_size*30,bin_size*30).astype(int))
ax2.set_xlabel('Absolute position (cm)')
ax2.set_ylabel('$\Delta$ F/F')
        
plt.savefig(os.path.join(savedst, f'rewardd_cell_{gc}_tuning_per_ep.svg'), bbox_inches='tight')

fig2,ax2 = plt.subplots(figsize=(5,5))
for ep in range(3):        
        ax2.plot(tcs_correct[ep,gc,:], label=f'rewloc {rewlocs[ep]}', color=colors[ep],linewidth=3)
        ax2.axvline(bins/2, color="k", linestyle='--',linewidth=3)
        
        ax2.spines[['top','right']].set_visible(False)
ax2.set_title(f'animal: {animal}, day: {day}\ncell # {gc}')
ax2.set_xticks(np.arange(0,bins+1,30))
ax2.set_xticklabels(np.round(np.arange(-np.pi, 
        np.pi+np.pi/1.5, np.pi/1.5),1))
ax2.set_xlabel('Radian position ($\Theta$)')
ax2.set_ylabel('$\Delta$ F/F')
plt.savefig(os.path.join(savedst, f'rewardd_cell_{gc}_aligned_tuning_per_ep.svg'), bbox_inches='tight')