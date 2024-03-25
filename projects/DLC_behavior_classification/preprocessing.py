import os, sys, shutil, tifffile, numpy as np, pandas as pd, scipy
from datetime import datetime
import scipy.io as sio, matplotlib.pyplot as plt, re
import h5py, pickle
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom your clone
sys.path.append(r'C:\Users\workstation2\Documents\MATLAB\han-lab') ## custom your clone
from utils.utils import listdir

def get_videos_from_hrz_csv(csvpth, dst, vidpth=r'\\storage1.ris.wustl.edu\ebhan\Active\new_eye_videos'):
    """
    csv = path of hrz vr behavior
    dst = where to store moved videos
    """
    df = pd.read_csv(csvpth, index_col=None)
    pths = [os.path.basename(xx) for xx in df.Var1.values]
            
    dates_with_monthname = re.findall(
        r'(\d{2}_(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)_\d{4})',
        str(pths)
    )
    mouse_names = [re.split(r"_", pth)[0].upper() for pth in pths]
    
    dates = []
    for s in dates_with_monthname:
        datetime_object = datetime.strptime(s[0], '%d_%b_%Y')
        dates.append(str(datetime_object.date()))    
    
    vids = [xx for xx in listdir(vidpth) if 'avi' in xx]
    vids2get= []
    for vid in vids:
        mnm = os.path.basename(vid)
        mouse_name = re.split(r"_", mnm)[1].upper() 
        date = str(datetime.strptime(re.split(r"_", mnm)[0], '%y%m%d').date())
        if 'avi' in mouse_name: mouse_name=mouse_name[:-4]
        if mouse_name in mouse_names and date in dates:
            print(vid, mouse_name, date)
            vids2get.append(vid)
            shutil.move(vid, os.path.join(dst, mnm))
            
    return vids2get

def match_eye_to_tail_videos(eyevids,dst, vidpth = r'\\storage1.ris.wustl.edu\ebhan\Active\tail_videos'):
    
    pths = [os.path.basename(xx) for xx in listdir(eyevids)]            
    mouse_names = [re.split(r"_", pth)[1].upper() for pth in pths]
    dates = [str(datetime.strptime(re.split(r"_", mnm)[0], '%y%m%d').date()) for mnm in pths]
    
    vids = [xx for xx in listdir(vidpth) if 'avi' in xx]
    vids2get= []
    for vid in vids:
        mnm = os.path.basename(vid)
        mouse_name = re.split(r"_", mnm)[1].upper() 
        date = str(datetime.strptime(re.split(r"_", mnm)[0], '%y%m%d').date())
        if 'avi' in mouse_name: mouse_name=mouse_name[:-4]
        for i,mnms in enumerate(mouse_names):
            if mnms==mouse_name and dates[i]==date:
                print(vid, mouse_name, date)
                vids2get.append(vid)                        
                shutil.move(vid, os.path.join(dst, mnm))
            
    return vids2get

def consecutive_stretch(x):
    z = np.diff(x)
    break_point = np.where(z != 1)[0]

    if len(break_point) == 0:
        return [x]

    y = [x[:break_point[0]]]
    for i in range(1, len(break_point)):
        y.append(x[break_point[i - 1] + 1:break_point[i]])
    y.append(x[break_point[-1] + 1:])
    
    return y 

def copyvr_dlc(vrdir, dlcfls): #TODO: find a way to do the same for clampex
    """copies vr files for existing dlc csvs
    essentially do not put a csv into this folder if you don't want it analysed

    Args:
        vrdir (_type_): _description_
        dlcfls (_type_): _description_

    Returns:
        _type_: _description_
    """
    csvs = [xx for xx in listdir(dlcfls) if '.csv' in xx]
    csvs.sort()
    mouse_data = {}
    mouse_dlc = []
    for csvfl in csvs:
        date = os.path.basename(csvfl)[:6]
        datetime_object = datetime.strptime(date, '%y%m%d')
        nm = os.path.basename(csvfl)[7:11]
        s = os.path.basename(csvfl)[7]
        # except for non 'e' leading mice
        if s.upper()!='E': nm = os.path.basename(csvfl)[7:10]
        if not nm in mouse_data.keys(): # allows for adding multiple dates
            mouse_data[nm] = [str(datetime_object.date())]
        else:
            mouse_data[nm].append(str(datetime_object.date()))
        mouse_dlc.append([nm,csvfl,str(datetime_object.date())])
    
    mice = list(np.unique(np.array(mouse_data.keys()))[0]) #make to simple list
    
    mouse_vr = []
    for mouse in mice:
        print(mouse)
        mouse = mouse.upper()
        vrfls = [xx for xx in listdir(vrdir, ifstring='.mat') if mouse in os.path.basename(xx)[:4].upper()]
        vrfls.sort()
        dates = mouse_data[mouse] # if a mouse has multiple dates        
        for xx in vrfls:
            s = re.findall(r'(\d{2}_(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)_\d{4})',
            xx)
            datetime_object = datetime.strptime(s[0][0], '%d_%b_%Y')                
            dt = str(datetime_object.date())
            if 'test'.lower() not in xx and 'test'.upper() not in xx and dt in dates:
                if not os.path.exists(os.path.join(dlcfls, os.path.basename(xx))): shutil.copy(xx,dlcfls)
                mouse_vr.append([mouse, xx, dt])
                
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
        savecsv = csv[:-4]+'_original.csv' # saves a copy of original
        df.to_csv(savecsv, index=None)
        cols=[[xx+"_x",xx+"_y",xx+"_likelihood"] for xx in pd.unique(df.iloc[0]) if xx!="bodyparts"]
        cols = [yy for xx in cols for yy in xx]; cols.insert(0, 'bodyparts')
        df.columns = cols
        df=df.drop([0,1])

        df.to_csv(csv)
    else:
        print("\n ******** please pass path to csv ********")
    return df

def interpolate_vrdata(uscanstop,uscanstart,dlcdf,vrdata):
    original_length = uscanstop-uscanstart
    target_length = len(dlcdf)
    x_original = np.linspace(0, 1, original_length)            
    # Creating the interpolation function
    f = scipy.interpolate.interp1d(x_original, vrdata, kind='linear')  # 'linear', 'nearest', 'cubic'..
    # Interpolating to get the new values
    x_int = np.linspace(0, 1, target_length)
    vrdata_interpolated = f(x_int)
    
    return vrdata_interpolated

def hdf5_to_dict(hdf5_object):
    """
    Recursively converts HDF5 groups and datasets into a nested dictionary.
    
    Parameters:
    - hdf5_object: An HDF5 file or group object.
    
    Returns:
    - A dictionary representation of the HDF5 file or group.
    """
    result = {}
    for key in hdf5_object.keys():
        # Check if the current object is a group or a dataset
        if isinstance(hdf5_object[key], h5py.Group):
            # If it's a group, recursively call the function
            result[key] = hdf5_to_dict(hdf5_object[key])
        else:
            # If it's a dataset, convert it to a NumPy array
            result[key] = hdf5_object[key][()]
    return result

def VRalign_automatic(vrfl, dlccsv, savedst, only_add_experiment=False):
    """zahra's implementation for VRstandendsplit for python dlc pipeline
    automatic alignment
    saves a png file with the imagesync start and stop so in the end you can corroborate
    and make sure recording doesnt have multiple imagesync chunks
    Args:
        vrfl (_type_): _description_
        dlccsv (_type_): _description_
    """
    if only_add_experiment:
        f = h5py.File(vrfl,'r')  #need to save vrfile with -v7.3 tag for this to work
        VR = f['VR']
        dst = os.path.join(savedst, os.path.basename(vrfl)[:16]+'_vr_dlc_align.p')
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
        dst = os.path.join(savedst, os.path.basename(vrfl)[:16]+'_vr_dlc_align.p')
        dlcdf = pd.read_csv(dlccsv)
        if 'bodyparts' not in dlcdf.columns.to_list(): #fixes messed up cols
            dlcdf = fixcsvcols(dlccsv) #saves over df
        if not os.path.exists(dst) and len(dlcdf)>0: # if pickle is made already
            print(f"******VR aligning {dst}******\n\n")            # aligns structure so size is the same
            f = h5py.File(vrfl,'r')  #need to save vrfile with -v7.3 tag for this to work
            VR = f['VR']
            
            # Find start and stop of imaging using VR
            imageSync = np.array([xx[0] for xx in VR['imageSync'][:]])
            # automatically estimate start and end of recordings
            # assumes there are no additional images after behavior
            # also prints out the imagesync plots if you want to do it manually?
            iinds = np.where(imageSync>1)
            uscanstart = np.min(iinds)
            uscanstop  = np.max(iinds)
            print(f'Start of scan: {uscanstart}')
            print(f'End of scan: {uscanstop}')
            print(f'Length of scan is {uscanstop - uscanstart}')
            print(f'Time of scan is {(VR["time"][:][uscanstop] - VR["time"][:][uscanstart])/60} minutes \n\n')
            inds = np.where(np.abs(np.diff(imageSync)) > 0.3 * np.max(np.abs(np.diff(imageSync))))[0]
            meaninds = np.mean(np.diff(inds))
            plt.figure()
            plt.subplot(3, 1, 1)
            plt.plot(imageSync)
            plt.plot(np.abs(np.diff(imageSync)) > 0.3 * np.max(np.abs(np.diff(imageSync))), 'r')
            plt.subplot(3, 1, 2)
            plt.plot(imageSync)
            plt.plot(np.abs(np.diff(imageSync)) > 0.3 * np.max(np.abs(np.diff(imageSync))), 'r')
            plt.xlim([inds[0] - 2.5 * meaninds, inds[0] + 2.5 * meaninds])
            plt.subplot(3, 1, 3)
            plt.plot(imageSync)
            plt.plot(np.abs(np.diff(imageSync)) > 0.3 * np.max(np.abs(np.diff(imageSync))), 'r')
            plt.xlim([inds[-1] - 4 * meaninds, inds[-1] + 2 * meaninds])
            plt.suptitle("Check to make sure imagesync var doesn't have multiple imaging sessions\n\
                Else needs manual alignment (regular func 'VRalign')")
            plt.savefig(os.path.join(vrfl[:-4]+"_imagesync_check.png"))
            # plt.close('all')
            scanstart = uscanstart
            scanstop = uscanstop
            check_imaging_start_before = 0  # there is no chance to recover 
            #imaging data from before you started VR so sets to 0

            # cuts all of the variables from VR
            # zahra's hack to fix vr variables after interpolation
            urewards = np.squeeze(VR['reward'][scanstart:scanstop]); 
            urewards_cs = np.zeros_like(urewards)
            urewards_cs = urewards==0.5
            urewards_cs = interpolate_vrdata(uscanstop,uscanstart,dlcdf,urewards_cs)            
            urewards=(urewards_cs>0).astype(int) # doesn't have reward variable anymore, so add that if needed (after 500msec)           
            uimageSync = np.squeeze(VR['imageSync'][scanstart:scanstop]); uimageSync = interpolate_vrdata(uscanstop,uscanstart,dlcdf,uimageSync); 
            uforwardvel = np.hstack(-0.013*VR['ROE'][scanstart:scanstop])/np.diff(np.squeeze(VR['time'][scanstart-1:scanstop]))
            uforwardvel = interpolate_vrdata(uscanstop,uscanstart,dlcdf,uforwardvel)
            uybinned = np.squeeze(VR['ypos'][scanstart:scanstop]); uybinned = interpolate_vrdata(uscanstop,uscanstart,dlcdf,uybinned); 
            unumframes = len(range(scanstart,scanstop))
            uVRtimebinned = np.squeeze(VR['time'][scanstart:scanstop] - check_imaging_start_before - VR['time'][scanstart])
            uVRtimebinned = interpolate_vrdata(uscanstop,uscanstart,dlcdf,uVRtimebinned) 
            utrialnum = np.squeeze(VR['trialNum'][scanstart:scanstop]); utrialnum = np.round(interpolate_vrdata(uscanstop,uscanstart,dlcdf,utrialnum))
            uchangeRewLoc = np.squeeze(VR['changeRewLoc'][scanstart:scanstop])
            uchangeRewLoc[0] = np.squeeze(VR['changeRewLoc'][0])
            uchangeRewLoc = np.round(interpolate_vrdata(uscanstop,uscanstart,dlcdf,uchangeRewLoc))
            ulicks = np.squeeze(VR['lick'][scanstart:scanstop])
            ulicks = interpolate_vrdata(uscanstop,uscanstart,dlcdf,ulicks); 
            # skip this for now, can binarize later with lick voltage
            # ulicks=ulicks>0
            ulickVoltage = np.squeeze(VR['lickVoltage'][scanstart:scanstop])             
            ulickVoltage = interpolate_vrdata(uscanstop,uscanstart,dlcdf,ulickVoltage)
            # utimedFF = np.linspace(0, (VR['time'][scanstop]-VR['time'][scanstart]), len(np.arange(uscanstart,uscanstop))) #subsample - then why are we doing this freq
            utimedFF = np.linspace(0, (VR['time'][scanstop]-VR['time'][scanstart]), len(dlcdf)) #subsample - then why are we doing this freq
            timedFF = utimedFF
            # interpolate instead!            
            rewards = urewards
            forwardvel = uforwardvel
            ybinned = uybinned
            trialnum = utrialnum
            changeRewLoc = uchangeRewLoc
            licks = ulicks
            lickVoltage = ulickVoltage
            
            # #initialize
            # rewards = np.zeros_like(timedFF)
            # forwardvel = np.zeros_like(timedFF)
            # ybinned = np.zeros_like(timedFF)
            # trialnum = np.zeros_like(timedFF)
            # changeRewLoc = np.zeros_like(timedFF)
            # licks = np.zeros_like(timedFF)
            # lickVoltage = np.zeros_like(timedFF)
                        
            colssave = [xx for xx in dlcdf.columns if 'bodyparts' not in xx and 'Unnamed' not in xx]

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
            # fix multiple frames per cs
            rewards_ = consecutive_stretch(np.where(rewards>0)[0])
            rewards_ = np.array([min(xx) for xx in rewards_])
            rewards = np.zeros_like(rewards)
            rewards[rewards_]=1 # assign cs to boolean
            
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
            vralign['experiment']=experiment
            vralign['ybinned']=np.hstack(ybinned)
            vralign['rewards']=np.hstack(rewards)
            vralign['forwardvel']=np.hstack(forwardvel)
            vralign['licks']=np.hstack(licks)
            vralign['changeRewLoc']=np.hstack(changeRewLoc)
            vralign['trialnum']=np.hstack(trialnum)
            vralign['timedFF']=np.hstack(timedFF)
            vralign['lickVoltage']=np.hstack(lickVoltage)
            for col in colssave:
                vralign[col] = dlcdf[col].values.astype(float)
            vralign['start_stop']=(uscanstart, uscanstop)
            # VR = hdf5_to_dict(VR)
            # vralign['VR']=VR
            # vralign['VR']=VR fails because h5py do not pickle apparently -_- 
            # saves dlc variables into pickle as well  
            print(list(vralign.keys()))
            with open(dst, "wb") as fp:   #Pickling
                pickle.dump(vralign, fp)            
            print(f"\n ********* saved to {dst} *********")

    return 

def VRalign(vrfl, dlccsv, savedst, only_add_experiment=False,mrzt=False):
    """zahra's implementation for VRstandendsplit for python dlc pipeline

    Args:
        vrfl (_type_): _description_
        dlccsv (_type_): _description_
    """
    if only_add_experiment:
        f = h5py.File(vrfl,'r')  #need to save vrfile with -v7.3 tag for this to work
        VR = f['VR']
        dst = os.path.join(savedst, os.path.basename(vrfl)[:16]+'_vr_dlc_align.p')
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
        dst = os.path.join(savedst, os.path.basename(vrfl)[:16]+'_vr_dlc_align.p')
        dlcdf = pd.read_csv(dlccsv)
        if 'bodyparts' not in dlcdf.columns.to_list(): #fixes messed up cols
            dlcdf = fixcsvcols(dlccsv) #saves over df
        if not os.path.exists(dst) and len(dlcdf)>0: # if pickle is made already
            print(f"******VR aligning {dst}******\n\n")            # aligns structure so size is the same
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
            print(f'Time of scan is {(VR["time"][:][uscanstop] - VR["time"][:][uscanstart])/60} minutes')

            plt.close('all')
            scanstart = uscanstart
            scanstop = uscanstop
            check_imaging_start_before = 0  # there is no chance to recover 
            #imaging data from before you started VR so sets to 0

            # cuts all of the variables from VR
            urewards = np.squeeze(VR['reward'][scanstart:scanstop]); 
            urewards_cs = np.zeros_like(urewards)
            urewards_cs = urewards==0.5
            urewards_cs = interpolate_vrdata(uscanstop,uscanstart,dlcdf,urewards_cs)
            urewards=(urewards_cs>0).astype(int) # doesn't have reward variable anymore, so add that if needed (after 500msec)           
            uimageSync = np.squeeze(VR['imageSync'][scanstart:scanstop]); uimageSync = interpolate_vrdata(uscanstop,uscanstart,dlcdf,uimageSync); 
            uforwardvel = np.hstack(-0.013*VR['ROE'][scanstart:scanstop])/np.diff(np.squeeze(VR['time'][scanstart-1:scanstop]))
            uforwardvel = interpolate_vrdata(uscanstop,uscanstart,dlcdf,uforwardvel)
            uybinned = np.squeeze(VR['ypos'][scanstart:scanstop]); uybinned = interpolate_vrdata(uscanstop,uscanstart,dlcdf,uybinned); 
            unumframes = len(range(scanstart,scanstop))
            uVRtimebinned = np.squeeze(VR['time'][scanstart:scanstop] - check_imaging_start_before - VR['time'][scanstart])
            uVRtimebinned = interpolate_vrdata(uscanstop,uscanstart,dlcdf,uVRtimebinned) 
            try:
                utrialnum = np.squeeze(VR['trialNum'][scanstart:scanstop]); utrialnum = np.round(interpolate_vrdata(uscanstop,uscanstart,dlcdf,utrialnum))
            except Exception as e:
                print('\n********** MRT VR align using trials var **********')
                utrialnum = np.squeeze(VR['trials'][scanstart:scanstop]); utrialnum = np.round(interpolate_vrdata(uscanstop,uscanstart,dlcdf,utrialnum))
            if not mrzt:
                uchangeRewLoc = np.squeeze(VR['changeRewLoc'][scanstart:scanstop])
                uchangeRewLoc[0] = np.squeeze(VR['changeRewLoc'][0])
                uchangeRewLoc = np.round(interpolate_vrdata(uscanstop,uscanstart,dlcdf,uchangeRewLoc))
                uchangeRewLoc_original = []
            else:
                try:
                    uchangeRewLoc = np.hstack(np.squeeze(VR['changeRewLoc']))
                    uchangeRewLoc= np.hstack(uchangeRewLoc)
                    uchangeRewLoc_original = np.hstack([np.ravel(VR[uchangeRewLoc[xx]][:]) for xx in range(len(uchangeRewLoc))]) # temp taking mean of all rew zones                
                except: #ifrewzones are not saved? fucked up save format
                    print('\n********** NOT saving changeRewLoc for VR! Cannot be imported from MRZT file **********')
                    uchangeRewLoc_original = np.zeros_like(uVRtimebinned)
            ulicks = np.squeeze(VR['lick'][scanstart:scanstop])
            ulicks = interpolate_vrdata(uscanstop,uscanstart,dlcdf,ulicks); ulicks=ulicks>0
            ulickVoltage = np.squeeze(VR['lickVoltage'][scanstart:scanstop])             
            ulickVoltage = interpolate_vrdata(uscanstop,uscanstart,dlcdf,ulickVoltage)
            # utimedFF = np.linspace(0, (VR['time'][scanstop]-VR['time'][scanstart]), len(np.arange(uscanstart,uscanstop))) #subsample - then why are we doing this freq
            utimedFF = np.linspace(0, (VR['time'][scanstop]-VR['time'][scanstart]), len(dlcdf)) #subsample - then why are we doing this freq
            timedFF = utimedFF
            #initialize
            rewards = np.zeros_like(timedFF)
            forwardvel = np.zeros_like(timedFF)
            ybinned = np.zeros_like(timedFF)
            trialnum = np.zeros_like(timedFF)
            changeRewLoc = np.zeros_like(timedFF)
            licks = np.zeros_like(timedFF)
            lickVoltage = np.zeros_like(timedFF)
                        
            colssave = [xx for xx in dlcdf.columns if 'bodyparts' not in xx and 'Unnamed' not in xx]
            print(colssave)    
            #downsample to vr frame rate
            # for cols in colssave:
            # # Generate an example array with 80000 elements
            #     arr = dlcdf[cols].values.astype(float)
            #     new_length = (uscanstop-uscanstart)
            #     # Calculate the downsampling factor
            #     factor = (uscanstop-uscanstart)/len(dlcdf)
            #     indices_to_keep = np.linspace(0, len(arr) - 1, num=new_length, dtype=int)
            #     # Select the elements based on the calculated indices
            #     downsampled_arr = arr[indices_to_keep]
            #     newdf[cols] = downsampled_arr

            for newindx in range(len(timedFF)): # this is longer in python lol
                if newindx%10000==0: print(newindx)
                if newindx == 0:
                    after = np.mean([timedFF[newindx], timedFF[newindx+1]])
                    rewards[newindx] = np.sum(urewards[uVRtimebinned <= after])
                    forwardvel[newindx] = np.mean(uforwardvel[uVRtimebinned <= after])
                    ybinned[newindx] = np.mean(uybinned[uVRtimebinned <= after])
                    trialnum[newindx] = np.max(utrialnum[uVRtimebinned <= after])
                    if mrzt:
                        changeRewLoc[newindx] = 0
                    else:
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
                    if mrzt:
                        changeRewLoc[newindx] = 0
                    else:
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
                    if mrzt:
                        changeRewLoc[newindx] = 0
                    else:
                        changeRewLoc[newindx] = np.sum(uchangeRewLoc[(uVRtimebinned>before) & (uVRtimebinned<=after)]);
            
            # fix multiple frames per cs
            rewards_ = consecutive_stretch(np.where(rewards>0)[0])
            rewards_ = np.array([min(xx) for xx in rewards_])
            rewards = np.zeros_like(rewards)
            rewards[rewards_]=1 # assign cs to boolean
            
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
            try: # need this to make it work with random reward vr
                ypossplit = np.where(np.diff(np.squeeze(ypos)) < -50)[0]
                # doing the same thing but with changerewloc
                rewlocsplit = np.where(changeRewLoc)[0]
                for c in range(1, len(rewlocsplit)): # 1 because the first is always the first index
                    if (rewlocsplit[c]-1 not in ypossplit):
                        idx = np.argmin(np.abs(ypossplit+1-rewlocsplit[c]))
                        changeRewLoc[ypossplit[idx]+1] = changeRewLoc[rewlocsplit[c]]
                        changeRewLoc[rewlocsplit[c]] = 0                                    
            except Exception as e:
                print(e)
                # fix string to int conversion when importing mat
            if not mrzt:
                experiment = str(bytes(np.ravel(VR['settings']['name']).tolist()))[2:-1]
            else:
                experiment = 'MultipleRewZoneTraining'
            vralign = {}
            vralign['experiment']=experiment
            vralign['ybinned']=np.hstack(ybinned)
            vralign['rewards']=np.hstack(rewards)
            vralign['forwardvel']=np.hstack(forwardvel)
            vralign['licks']=np.hstack(licks)
            vralign['changeRewLoc']=np.hstack(changeRewLoc)
            vralign['trialnum']=np.hstack(trialnum)
            vralign['timedFF']=np.hstack(timedFF)
            vralign['lickVoltage']=np.hstack(lickVoltage)
            if len(uchangeRewLoc_original)>0:
                vralign['uchangeRewLoc_original'] = uchangeRewLoc_original
            for col in colssave:
                vralign[col] = dlcdf[col].values.astype(float)
            vralign['start_stop']=(uscanstart, uscanstop)
            # VR = hdf5_to_dict(VR)
            # vralign['VR']=VR
            # vralign['VR']=VR fails because h5py do not pickle apparently -_- 
            # saves dlc variables into pickle as well  
            print(list(vralign.keys()))
            with open(dst, "wb") as fp:   #Pickling
                pickle.dump(vralign, fp)            
            print(f"\n ********* saved to {dst} *********")

    return 