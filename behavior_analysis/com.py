import numpy as np, pickle, scipy
import matplotlib.pyplot as plt

import numpy as np

def correct_artifact_licks(ybinned, licks):
    """
    Correct artifact licks by removing consecutive licks and licks occurring before a certain position.
    
    Parameters:
    ybinned (numpy.ndarray): Array of positions corresponding to licks.
    licks (numpy.ndarray): Array of licks (binary 0 or 1).
    
    Returns:
    numpy.ndarray: Corrected array of licks.
    """
    x = 3  # Modify the cm threshold as needed

    # Calculate the difference between consecutive licks
    diffL = np.diff(licks) == 1
    
    # Pad a zero at the beginning
    diffL = np.insert(diffL, 0, 0)
    
    # Keep only the starting point of the lick transients
    licks = licks * diffL
    
    # Delete all the licks before 'x' cm
    licks[ybinned <= x] = 0
    
    return licks


def analyze_mouse_behavior(VR):

    # EH variables 
    rewWinNeg = 10  # EH -cm from rewStart peri-reward window
    rewWinPos = 5   # EH +cm peri-reward window
    if 'settings' in VR:
        rewSize = np.hstack(VR['settings']['rewardZone'])[0][0][0] / 2  # EH 15cm rew zone + -
    else:
        rewSize = 15 / 2
    numTrialsStart = 5  # EH number of trials to average at start and end of rew epoch
    numTrialsEnd = 5    # EH number of trials to average at start and end of rew epoch

    # GM Variables
    speedbinsize = 3
    conversion = -0.013
    slidingwindow = 5
    testlearningposition = 60  # cm
    
    changeRewLoc = np.where(VR['changeRewLoc'][0])[0]  # find the change in rew locations
    changeRewLoc = np.append(changeRewLoc, len(VR['changeRewLoc'][0]))  # vector containing where each reward location ends

    RewLoc = VR['changeRewLoc'][0][changeRewLoc[:-1]]  # find the rew locations (in cm)
    RewLocStart = RewLoc - rewSize  # find the start of rew locations (in cm) %EH

    VR['lick'] = correct_artifact_licks(VR['ypos'][0], VR['lick'][0])  # correct the lick artifacts
    
    # Delete large ROE spikes if missed by VR
    VR['ROE'] = VR['ROE'][0]
    for ll in range(1, len(VR['ROE'])):
        if (VR['ROE'][ll] - VR['ROE'][ll - 1]) <= -45 and VR['ROE'][ll] < -5 and VR['ROE'][ll - 1] < -5:
            VR['ROE'][ll] = VR['ROE'][ll - 1]

    # Convert ROE to Speed
    VR['ROE'] = (conversion * VR['ROE']) / np.append(0, np.diff(VR['time']))
    
    # Pre-allocated variables
    num_epochs = len(changeRewLoc) - 1
    COM = [[] for _ in range(num_epochs)]
    stdCOM = [[] for _ in range(num_epochs)]
    allCOM = [[] for _ in range(num_epochs)]
    allstdCOM = [[] for _ in range(num_epochs)]
    ROE = [[] for _ in range(num_epochs)]
    meanROE = [[] for _ in range(num_epochs)]
    stdROE = [[] for _ in range(num_epochs)]
    ROEcount = [[] for _ in range(num_epochs)]
    binypos = [[] for _ in range(num_epochs)]
    binstarted = [[] for _ in range(num_epochs)]
    slidingMeanCOM = [[] for _ in range(num_epochs)]
    slidingVarCOM = [[] for _ in range(num_epochs)]
    cutROE = [[] for _ in range(num_epochs)]
    meancutROE = [[] for _ in range(num_epochs)]
    semcutROE = [[] for _ in range(num_epochs)]
    timecount = [[] for _ in range(num_epochs)]
    InterLickInterval = []

    allRatio = [[] for _ in range(num_epochs)]  # EH ratio of peri-reward licks for all trials for each rew location
    
    if np.sum(VR['lick'] > 0) == 0:
        print('This session has no licks')
        return
    VR['reward']=VR['reward'][0]
    VR['ypos']=VR['ypos'][0]
    for kk in range(num_epochs):       
        trialNum = VR['trialNum'][0]        
        numProbe = 3  # same as numprobe from runtime code
        
        rewStart = RewLoc[kk] - rewSize  # EH define peri-reward window
        periLow = rewStart - rewWinNeg   # EH lower end of window
        periHigh = rewStart + rewWinPos  # EH upper end of window

        difftrials = np.diff(trialNum[changeRewLoc[kk]:changeRewLoc[kk+1]])
        difftrials = np.insert(difftrials, 0, 0)
        
        if trialNum[0] == numProbe and kk == 0:
            difftrials[0] = 1
            
        startnotprobe = np.where(trialNum[changeRewLoc[kk]:] > (numProbe - 1))[0][0] + changeRewLoc[kk]  # find the starting of each reward location after the probe trials
        
        trials = np.where(difftrials >= 1)[0] + changeRewLoc[kk]  # find the starting of each trial within reward location

        for jj in range(len(trials) - 1):
            trial_slice = slice(trials[jj] + 1, trials[jj + 1])
            if np.sum(VR['reward'][trial_slice]) > 0 and np.sum(VR['lick'][trial_slice]) > 0:  # if it's a successful trial
                licking = np.where(VR['lick'][trial_slice])[0] + trials[jj]  # find all the licks before and equal to the reward lick
                licking = licking[np.diff(np.append(0, licking)) != 1]  # remove consecutive licks
                
                periLick = (VR['ypos'][licking] > periLow) & (VR['ypos'][licking] < periHigh)  # EH logical of peri-reward licks
                ratio = np.sum(periLick) / len(periLick)  # EH ratio of peri to total licks for trial
                
                COM[kk].append(np.mean(VR['ypos'][licking]) - RewLoc[kk])  # calculate the normalized COM for each trial and store it
                stdCOM[kk].append(np.std(VR['ypos'][licking]) - RewLoc[kk])  # calculate the std of the COM for each trial and store it
                
                lickDist = np.abs(VR['ypos'][licking] - RewLocStart[kk])  # EH abs distance of each lick from rewStart
                
                start = np.where(np.diff(VR['ypos'][trial_slice]))[0][0] + trials[jj]  # defines start and stop indices for binning
                stop = np.where(VR['reward'][trial_slice])[0][0] + trials[jj]
                
                binstart = int(np.floor(VR['ypos'][start]))
                binstop = int(np.ceil(VR['ypos'][stop]))
                
                failure = 0  # keep track of failed trials
                
            else:
                licking = np.where(VR['lick'][trial_slice])[0] + trials[jj]  # else, find all the licks of that trial
                licking = licking[np.diff(np.append(0, licking)) != 1]  # remove consecutive licks
                lickDist = np.abs(VR['ypos'][licking] - RewLocStart[kk])  # EH abs distance of each lick from rewStart
                
                failure = 1  # mark as failure
                
                periLick = (VR['ypos'][licking] > periLow) & (VR['ypos'][licking] < periHigh)  # EH logical of peri-reward licks
                ratio = np.sum(periLick) / len(periLick)  # EH ratio of peri to total licks for trial

            # Additional data processing for GM InterLickInterval, Velocity, etc.

        # Implement sliding window analysis on COM data

    # Finalize the raw data plot for the session

    # The function doesn't return anything, but you can add return statements as needed to extract specific data.