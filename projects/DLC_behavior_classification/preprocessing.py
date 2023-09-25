import os, sys, shutil, tifffile, numpy as np, pandas as pd
from datetime import datetime
import scipy.io as sio, matplotlib.pyplot as plt
import h5py, pickle
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom your clone
from utils.utils import listdir

def copyvr_dlc(vrdir, dlcfls): #TODO: find a way to do the same for clampex
    
    csvs = [xx for xx in listdir(dlcfls) if '.csv' in xx]
    csvs.sort()
    mouse_data = {}
    mouse_dlc = []
    for csvfl in csvs:
        print(csvfl)        
        date = os.path.basename(csvfl)[:6]
        datetime_object = datetime.strptime(date, '%y%m%d')
        if not os.path.basename(csvfl)[7:11] in mouse_data.keys(): # allows for adding multiple dates
            mouse_data[os.path.basename(csvfl)[7:11]] = [str(datetime_object.date())]
        else:
            mouse_data[os.path.basename(csvfl)[7:11]].append(str(datetime_object.date()))
        mouse_dlc.append([os.path.basename(csvfl)[7:11],csvfl,str(datetime_object.date())])
    
    mice = list(np.unique(np.array(mouse_data.keys()))[0]) #make to simple list
    mouse_vr = []
    for mouse in mice:
        print(mouse)
        mouse = mouse.upper()
        vrfls = [xx for xx in listdir(vrdir, ifstring='.mat') if mouse in os.path.basename(xx)[:4].upper()]
        vrfls.sort()
        dates = mouse_data[mouse] # if a mouse has multiple dates
        for xx in vrfls:
            if 'test'.lower() not in xx and 'test'.upper() not in xx and str(datetime.strptime(os.path.basename(xx)[5:16], 
                '%d_%b_%Y').date()) in dates:
                shutil.copy(xx,dlcfls)
                mouse_vr.append([mouse, xx, str(datetime.strptime(os.path.basename(xx)[5:16], 
                '%d_%b_%Y').date())])
        print(f"\n********* copied vr files to dlc pose data for {mouse} *********")
    
    # pair dlc files with vr
    paired_df = []
    for mouse, csv, exp_date in mouse_dlc:
        ind = np.where((exp_date==np.array(mouse_vr)[:,2]) & (mouse==np.array(mouse_vr)[:,0]))[0][0]
        # remove leading path in case we move things
        paired_df.append([mouse, os.path.basename(csv), os.path.basename(np.array(mouse_vr)[ind,1]), exp_date])
    mouse_df = pd.DataFrame(paired_df, columns = ["mouse", "DLC", "VR", "date"])

    return mouse_df

def fixcsvcols(csv):
    if type(csv) == str:
        df = pd.read_csv(csv)
        cols=[[xx+"_x",xx+"_y",xx+"_likelihood"] for xx in pd.unique(df.iloc[0]) if xx!="bodyparts"]
        cols = [yy for xx in cols for yy in xx]; cols.insert(0, 'bodyparts')
        df.columns = cols
        df=df.drop([0,1])

        df.to_csv(csv)
    else:
        print("\n ******** please pass path to csv ********")
    return df

def VRalign(vrfl, dlccsv, only_add_experiment=False):
    """zahra's implementation for VRstandendsplit for python dlc pipeline
    TODO: does not care about planes, figure out what to do with this
    NOTE: subsamples to half of video (imaging frames) - should I not do this???

    Args:
        vrfl (_type_): _description_
        dlccsv (_type_): _description_
    """
    if only_add_experiment:
        f = h5py.File(vrfl,'r')  #need to save vrfile with -v7.3 tag for this to work
        VR = f['VR']
        dst = os.path.join(os.path.dirname(vrfl),
            os.path.basename(vrfl)[:16]+'_vr_dlc_align.p')
        with open(dst, "rb") as fp: #unpickle
            vralign = pickle.load(fp)
        # fix string to int conversion when importing mat
        experiment = str(bytes(np.ravel(VR['settings']['name']).tolist()))[2:-1]        
        vralign['experiment'] = experiment
        with open(dst, "wb") as fp:   #Pickling
            pickle.dump(vralign, fp)
        print(f"\n ********* saved to {dst} *********")
    else:
        # modified VRstartendsplit
        dst = os.path.join(os.path.dirname(vrfl),
                os.path.basename(vrfl)[:16]+'_vr_dlc_align.p')
        print(dst)
        if not os.path.exists(dst): # if pickle is made already
            f = h5py.File(vrfl,'r')  #need to save vrfile with -v7.3 tag for this to work
            VR = f['VR']
            
            # Find start and stop of imaging using VR
            imageSync = np.array([xx[0] for xx in VR['imageSync'][:]])

            # find difference of imagesync and use that to find beginning and end of script
            inds = np.where(np.abs(np.diff(imageSync)) > 0.3 * np.max(np.abs(np.diff(imageSync))))[0]
            meaninds = np.mean(np.diff(inds))
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(imageSync)
            plt.plot(np.abs(np.diff(imageSync)) > 0.3 * np.max(np.abs(np.diff(imageSync))), 'r')
            plt.subplot(2, 1, 2)
            plt.plot(imageSync)
            plt.plot(np.abs(np.diff(imageSync)) > 0.3 * np.max(np.abs(np.diff(imageSync))), 'r')
            plt.xlim([inds[0] - 2.5 * meaninds, inds[0] + 2.5 * meaninds])
            uscanstart = plt.ginput(1)
            uscanstart = int(round(uscanstart[0][0]))
            plt.close()

            plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(imageSync)
            plt.plot(np.abs(np.diff(imageSync)) > 0.3 * np.max(np.abs(np.diff(imageSync))), 'r')
            plt.subplot(2, 1, 2)
            plt.plot(imageSync)
            plt.plot(np.abs(np.diff(imageSync)) > 0.3 * np.max(np.abs(np.diff(imageSync))), 'r')
            plt.xlim([inds[-1] - 4 * meaninds, inds[-1] + 2 * meaninds])
            uscanstop  = plt.ginput(1)
            uscanstop = int(round(uscanstop[0][0]))
            print(f'Length of scan is {uscanstop - uscanstart}')
            print(f'Time of scan is {VR["time"][:][uscanstop] - VR["time"][:][uscanstart]}')

            plt.close('all')
            if 'imageSync' not in VR:  # if there was no VR.imagesync, rewrites scanstart and scanstop to be in VR iteration indices
                # buffer = diff(data[:,4])
                VRlastlick = np.where(VR['lick'][:] > 0)[0][-1]
                abflicks = np.where(np.diff(VR['data'][:, 2]) > 0)[0]
                buffer = abflicks[-1] / 1000 - VR['time'][:][0][VRlastlick]
                check_imaging_start_before = uscanstart / 1000 - buffer  # there is a chance to recover imaging data from before you started VR (if you made an error) in this case so checking for that
                scanstart = np.argmin(np.abs(VR['time'][:][0] - (uscanstart / 1000 - buffer)))
                scanstop = np.argmin(np.abs(VR['time'][:][0] - (uscanstop / 1000 - buffer)))
            else:
                scanstart = uscanstart
                scanstop = uscanstop
                check_imaging_start_before = 0  # there is no chance to recover 
                #imaging data from before you started VR so sets to 0

            # cuts all of the variables from VR
            urewards = np.squeeze(VR['reward'][scanstart:scanstop])
            uimageSync = np.squeeze(VR['imageSync'][scanstart:scanstop])
            uforwardvel = -0.013*VR['ROE'][scanstart:scanstop]/np.diff(np.squeeze(VR['time'][scanstart-1:scanstop]))
            uybinned = np.squeeze(VR['ypos'][scanstart:scanstop])
            unumframes = len(range(scanstart,scanstop))
            uVRtimebinned = np.squeeze(VR['time'][scanstart:scanstop] - check_imaging_start_before - VR['time'][scanstart])
            utrialnum = np.squeeze(VR['trialNum'][scanstart:scanstop])
            uchangeRewLoc = np.squeeze(VR['changeRewLoc'][scanstart:scanstop])
            uchangeRewLoc[0] = np.squeeze(VR['changeRewLoc'][0])
            ulicks = np.squeeze(VR['lick'][scanstart:scanstop])
            ulickVoltage = np.squeeze(VR['lickVoltage'][scanstart:scanstop])
            
            # aligns structure so size is the same GM
            dlcdf = pd.read_csv(dlccsv)
            if 'bodyparts' not in dlcdf.columns.to_list(): #fixes messed up cols
                dlcdf = fixcsvcols(dlccsv) #saves over df
            utimedFF = np.linspace(0, (VR['time'][scanstop]-VR['time'][scanstart]), 
                        round((len(dlcdf)/2))) #subsample - then why are we doing this freq
            timedFF = utimedFF
            #initialize
            rewards = np.zeros_like(timedFF)
            forwardvel = np.zeros_like(timedFF)
            ybinned = np.zeros_like(timedFF)
            trialnum = np.zeros_like(timedFF)
            changeRewLoc = np.zeros_like(timedFF)
            licks = np.zeros_like(timedFF)
            lickVoltage = np.zeros_like(timedFF)

            for newindx in range(len(timedFF)): # this is longer in python lol
                if newindx%10000==0: print(newindx)
                if newindx == 0:
                    after = np.mean([timedFF[newindx], timedFF[newindx+1]])
                    rewards[newindx] = np.sum(urewards[uVRtimebinned <= after])
                    forwardvel[newindx] = np.mean(uforwardvel[uVRtimebinned <= after])
                    ybinned[newindx] = np.mean(uybinned[uVRtimebinned <= after])
                    trialnum[newindx] = np.max(utrialnum[uVRtimebinned <= after])
                    changeRewLoc[newindx] = uchangeRewLoc[newindx]
                    licks[newindx] = np.sum(ulicks[uVRtimebinned <= after]) > 0
                    lickVoltage[newindx] = np.mean(ulickVoltage[uVRtimebinned <= after])
                    
                elif newindx == len(timedFF)-1:
                    before = np.mean([timedFF[newindx], timedFF[newindx-1]])
                    rewards[newindx] = np.sum(urewards[uVRtimebinned > before])
                    forwardvel[newindx] = np.mean(uforwardvel[uVRtimebinned > before])
                    ybinned[newindx] = np.mean(uybinned[uVRtimebinned > before])
                    trialnum[newindx] = np.max(utrialnum[uVRtimebinned > before],
                                    initial=0)
                    changeRewLoc[newindx] = np.sum(uchangeRewLoc[uVRtimebinned > before],
                                    initial=0)
                    licks[newindx] = np.sum(ulicks[uVRtimebinned > 0])
                    lickVoltage[newindx] = np.mean(ulickVoltage[uVRtimebinned > before])

                else:                                                      
                    before = np.mean([timedFF[newindx], timedFF[newindx-1]])
                    after = np.mean([timedFF[newindx], timedFF[newindx+1]])
                    #idk what these conditions are for
                    if sum((uVRtimebinned>before) & (uVRtimebinned<=after))==0 and after<=check_imaging_start_before:
                        rewards[newindx] = urewards[0]
                        licks[newindx] = ulicks[0]
                        ybinned[newindx] = uybinned[0]
                        forwardvel[newindx] = forwardvel[0]
                        changeRewLoc[newindx] = 0
                        trialnum[newindx] = utrialnum[0]
                        lickVoltage[newindx] = ulickVoltage[newindx]
                    elif sum((uVRtimebinned>before) & (uVRtimebinned<=after))==0 and after>check_imaging_start_before:
                        rewards[newindx] = rewards[newindx-1]
                        licks[newindx] = licks[newindx-1]
                        ybinned[newindx] = ybinned[newindx-1]
                        forwardvel[newindx] = forwardvel[newindx-1]
                        changeRewLoc[newindx] = 0
                        trialnum[newindx] = trialnum[newindx-1]
                        lickVoltage[newindx] = ulickVoltage[newindx-1]
                    else: # probably take longer bc of vector wise and
                        rewards[newindx] = np.sum(urewards[(uVRtimebinned>before) & (uVRtimebinned<=after)])
                        licks[newindx] = np.sum(ulicks[(uVRtimebinned>before) & (uVRtimebinned<=after)])>0
                        lickVoltage[newindx] = np.mean(ulickVoltage[(uVRtimebinned>before) & (uVRtimebinned<=after)])
                        # try:
                        if np.min(np.diff(uybinned[(uVRtimebinned>before) & (uVRtimebinned<=after)]),initial=0) < -50: # added initial cond bc min is allow on zero arrays in matlab
                            dummymin =  np.min(uybinned[(uVRtimebinned>before) & (uVRtimebinned<=after)])
                            dummymax = np.max(uybinned[(uVRtimebinned>before) & (uVRtimebinned<=after)])
                            dummymean = np.mean(uybinned[(uVRtimebinned>before) & (uVRtimebinned<=after)])
                            ybinned[newindx] = ((dummymean/(dummymax-dummymin))<0.5)*dummymin+((dummymean/(dummymax-dummymin))>=0.5)*dummymax
                            dummytrialmin =  np.min(utrialnum[(uVRtimebinned>before) & (uVRtimebinned<=after)])
                            dummytrialmax = np.max(utrialnum[(uVRtimebinned>before) & (uVRtimebinned<=after)])
                            dummytrialmean = np.mean(utrialnum[(uVRtimebinned>before) & (uVRtimebinned<=after)])
                            trialnum[newindx] = ((dummytrialmean/(dummytrialmax-dummytrialmin))<0.5)*dummytrialmin+((dummytrialmean/(dummytrialmax-dummytrialmin))>=0.5)*dummytrialmax
                        else:
                            ybinned[newindx] = np.mean(uybinned[(uVRtimebinned>before) & (uVRtimebinned<=after)])
                            trialnum[newindx] = np.max(utrialnum[(uVRtimebinned>before) & (uVRtimebinned<=after)])
                        # except Exception as e:
                        #     print(f"\n some issue with selecting ybinned in early time points? id {newindx}")
                        
                    forwardvel[newindx] = np.mean(uforwardvel[(uVRtimebinned>before) & (uVRtimebinned<=after)]);
                    changeRewLoc[newindx] = np.sum(uchangeRewLoc[(uVRtimebinned>before) & (uVRtimebinned<=after)]);
            
            # sometimes trial number increases by 1 for 1 frame at the end of an epoch before
            # going to probes. this removes those
            trialchange = np.concatenate(([0], np.diff(np.squeeze(trialnum))))
            # GM and ZD added to fix times when VR does not have data for the imaging
            # frames; seems to happen randomly
            artefact1 = np.where(np.concatenate(([0, 0], trialchange[:-2])) == 1 & (trialchange < 0))[0]
            trialnum[artefact1-1] = trialnum[artefact1];
            artefact = np.where(np.concatenate(([0], trialchange[:-1])) == 1 & (trialchange < 0))[0]
            if artefact.size != 0:
                trialnum[artefact-1] = trialnum[artefact-2]
                            
            # this ensures that all trial number changes happen on when the
            # yposition goes back to the start, not 1 frame before or after
            ypos = ybinned.copy()
            trialsplit = np.where(np.diff(np.squeeze(trialnum)))[0]
            ypossplit = np.where(np.diff(np.squeeze(ypos)) < -50)[0]
            #ZD commented out because it was setting trialnum to a constant as
            # previously debugged by GM and ZD above
            # for t in range(len(trialsplit)):
            #     try: # accounts for different lengths, ok to bypass?
            #         if trialsplit[t] < ypossplit[t]:
            #             trialnum[trialsplit[t]:ypossplit[t]+1] = trialnum[trialsplit[t]-1]
            #         elif trialsplit[t] > ypossplit[t]:
            #             trialnum[ypossplit[t]+1:trialsplit[t]] = trialnum[trialsplit[t]+1]
            #     except IndexError:
            #         pass
                
            # doing the same thing but with changerewloc
            rewlocsplit = np.where(changeRewLoc)[0]
            for c in range(1, len(rewlocsplit)): # 1 because the first is always the first index
                if (rewlocsplit[c]-1 not in ypossplit):
                    idx = np.argmin(np.abs(ypossplit+1-rewlocsplit[c]))
                    changeRewLoc[ypossplit[idx]+1] = changeRewLoc[rewlocsplit[c]]
                    changeRewLoc[rewlocsplit[c]] = 0
            
            # fix string to int conversion when importing mat
            experiment = str(bytes(np.ravel(VR['settings']['name']).tolist()))[2:-1]
            vralign = {}
            vralign['experiment'] = experiment
            vralign['ybinned']=ybinned
            vralign['rewards']=rewards
            vralign['forwardvel']=forwardvel
            vralign['licks']=licks
            vralign['changeRewLoc']=changeRewLoc
            vralign['trialnum']=trialnum
            vralign['timedFF']=timedFF
            vralign['lickVoltage']=lickVoltage    
            with open(dst, "wb") as fp:   #Pickling
                pickle.dump(vralign, fp)
            print(f"\n ********* saved to {dst} *********")

    return 