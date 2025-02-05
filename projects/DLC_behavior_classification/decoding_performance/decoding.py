"""
decoding
use pupil and lick quantile
"""
import numpy as np, pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import pickle, os, sys, matplotlib.pyplot as plt, matplotlib as mpl
import numpy as np, scipy
from mpl_toolkits.axes_grid1 import make_axes_locatable
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 6
mpl.rcParams["ytick.major.size"] = 6
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"

def consecutive_stretch(x):
    z = np.diff(x)
    break_point = np.where(z != 1)[0]
    
    if len(break_point) == 0:
        return [x]
    
    y = []
    prev_idx = 0
    for idx in break_point:
        y.append(x[prev_idx:idx+1])
        prev_idx = idx + 1
    
    if prev_idx < len(x):
        y.append(x[prev_idx:])
    
    return y

def calculate_ALR(licks_cue, duration_cue, licks_baseline, duration_baseline):
    # Calculate lick rates
    lick_rate_cue = licks_cue / duration_cue
    lick_rate_baseline = licks_baseline / duration_baseline
    
    # Calculate ALR
    alr = lick_rate_cue / lick_rate_baseline
    
    return alr

# Example data
licks_cue = 10
duration_cue = 30  # seconds
licks_baseline = 5
duration_baseline = 30  # seconds

# Calculate ALR
alr = calculate_ALR(licks_cue, duration_cue, licks_baseline, duration_baseline)

def get_lick_ratio_per_trial(eps, trialnum, rewards, areas_all, 
        licks, rewlocs, ybinned, Fs):
    """
    Function to quantify basic behavior for HRZ.

    Args:
        trialnum (list or np.array): Trial numbers
        reward (list or np.array): Reward values (0 or 1)

    Returns:
        success (int): Number of successful trials
        fail (int): Number of failed trials
        str (list): List of successful trial numbers
        ftr (list): List of failed trial numbers
        ttr (list): List of valid trial numbers (excluding probe trials)
        total_trials (int): Total number of valid trials
    """    
    lick_ratio_ep = []; areas_ep_binned = []
    for ep in range(len(eps)-1):
        rewloc = rewlocs[ep]
        trials = trialnum[eps[ep]:eps[ep+1]]
        reward = rewards[eps[ep]:eps[ep+1]]
        areas = areas_all[eps[ep]:eps[ep+1]]
        ypos = ybinned[eps[ep]:eps[ep+1]]
        lick = licks[eps[ep]:eps[ep+1]]
        lick_ratio = np.zeros_like(np.unique(trials[trials>=3]))
        areas_trial_binned = np.zeros_like(np.unique(trials[trials>=3]))
        for tt,trial in enumerate(np.unique(trials[trials>=3])):
            if trial >= 3:  # Exclude probe trials (trial < 3)
                areas_trial = areas[trials==trial]            
                lick_trial = lick[trials==trial]    
                areas_trial_binned[tt]=np.nanmean(areas_trial[ypos[trials==trial]<rewloc]) # pre reward
                if sum(reward[trials == trial] == 1)>0:
                    lick_ratio[tt] = np.ceil(sum(lick_trial[(ypos[trials==trial]>rewloc-5) & (ypos[trials==trial]<ypos[trials==trial][reward[trials == trial] == 1][0])])/31.25)/(sum(lick_trial)/31.25)
                else:
                    lick_ratio[tt] = lick_trial[(ypos[trials==trial]>rewloc-5) & (ypos[trials==trial]<rewloc+5)]
        lick_ratio_ep.append(lick_ratio)
        areas_ep_binned.append(areas_trial_binned)
        
    return lick_ratio_ep, areas_ep_binned


def get_trial_binary(eps, trialnum, rewards, areas_all, rewlocs, ybinned):
    """
    Function to quantify basic behavior for HRZ.

    Args:
        trialnum (list or np.array): Trial numbers
        reward (list or np.array): Reward values (0 or 1)

    Returns:
        success (int): Number of successful trials
        fail (int): Number of failed trials
        str (list): List of successful trial numbers
        ftr (list): List of failed trial numbers
        ttr (list): List of valid trial numbers (excluding probe trials)
        total_trials (int): Total number of valid trials
    """    
    trial_ep_binary = []; areas_ep_binned = []
    for ep in range(len(eps)-1):
        rewloc = rewlocs[ep]
        trials = trialnum[eps[ep]:eps[ep+1]]
        reward = rewards[eps[ep]:eps[ep+1]]
        areas = areas_all[eps[ep]:eps[ep+1]]
        ypos = ybinned[eps[ep]:eps[ep+1]]
        trials_binary = np.zeros_like(np.unique(trials[trials>=3]))
        areas_trial_binned = np.zeros_like(np.unique(trials[trials>=3]))
        for tt,trial in enumerate(np.unique(trials[trials>=3])):
            if trial >= 3:  # Exclude probe trials (trial < 3)
                if np.sum(reward[trials == trial] == 1) > 0:  # If reward was found in the trial
                    trials_binary[tt]=1
                areas_trial = areas[trials==trial]                
                areas_trial_binned[tt]=np.nanmean(areas_trial[ypos[trials==trial]<rewloc]) # pre reward
        trial_ep_binary.append(trials_binary)
        areas_ep_binned.append(areas_trial_binned)
        
    return trial_ep_binary, areas_ep_binned

# Load your dataset
# Assuming your dataset has columns for 'pupil_size', 'velocity', and 'successful_trial'
pdst = r"I:\vids_to_analyze\face_and_pupil\pupil\E186_23_Dec_2022_vr_dlc_align.p"
with open(pdst, "rb") as fp: #unpickle
        vralign = pickle.load(fp)
# hacking changerewloc
eps = consecutive_stretch(np.where(vralign['changeRewLoc'])[0])
eps = [min(xx) for xx in eps]
eps = np.array(eps[1:])[np.diff(eps)>1000]
eps = np.append(eps,len(vralign['changeRewLoc']))
if not 0 in eps:
    eps = np.append(eps,0)
eps = np.sort(eps)
rewards = vralign['rewards']
ybinned = vralign['ybinned']
licks = vralign['licks']
rewlocs = [137,101,70,109]
trbin, abin = get_trial_binary(eps, vralign['trialnum'], rewards, vralign['areas_residual'], rewlocs,
                    ybinned)
x = np.array(np.hstack(abin[0]))
scaler = MinMaxScaler(feature_range=(0, 1))
# normalize
X = x[~np.isnan(x)].reshape(-1, 1)
X = scaler.fit_transform(X)
y =  np.array(np.hstack(trbin[0]))
y = y[~np.hstack(np.isnan(x))]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01)

# Train a Gaussian Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Function to predict the outcome of the next trial using Bayesian decoder
def predict_next_trial(pupil_size):
    features = np.array([pupil_size])
    probs = model.predict_proba(features)
    prob_success = probs[0][1]
    prob_failure = probs[0][0]
    
    if prob_success > prob_failure:
        print(f"The next trial is predicted to be successful with probability {prob_success:.2f}.")
        return 1
    else:
        print(f"The next trial is predicted to be unsuccessful with probability {prob_failure:.2f}.")
        return 0

# Example usage
test = abin[2]
Xt = test[~np.isnan(test)].reshape(-1,1)
Xt = scaler.fit_transform(Xt)
predictions = [predict_next_trial(xx) for xx in Xt] 
plt.plot(predictions)
plt.plot(trbin[2][np.hstack(~np.isnan(test))])
    