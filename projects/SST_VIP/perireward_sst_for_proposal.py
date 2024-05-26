import numpy as np, sys, os, scipy.io
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.DLC_behavior_classification import eye
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"

# Helper functions (equivalent to MATLAB functions)
def load_mat_file(filepath):
    return scipy.io.loadmat(filepath)

# Constants and initial setup
an = 'e136'
dy = 4
base_path = "Y:/analysis/fmats"
mat_file_template = os.path.join(base_path, an, f"{an}_day{dy:03d}_plane{{}}_Fall.mat")

# Load data
pln0 = load_mat_file(mat_file_template.format(0))
pln1 = load_mat_file(mat_file_template.format(1))
pln2 = load_mat_file(mat_file_template.format(2))

plns = [pln0, pln1, pln2]
planes = 3
binsize = 0.2  # seconds
range_val = 8  # seconds

changeRewLoc = pln0['changeRewLoc'][0]
eps = np.where(changeRewLoc > 0)[0]
eps = np.append(eps, len(changeRewLoc))
track_length = 180  # cm
nbins = int(track_length / bin_size)
rewlocs = changeRewLoc[changeRewLoc > 0]
rewsize = 15  # cm

savedst = 'C:/Users/Han/Box/neuro_phd_stuff/han_2023-/figure_data/sst'
if not os.path.exists(savedst):
    os.makedirs(savedst)

dFF_all_planes = [pln0['dFF_iscell'], pln1['dFF_iscell'], pln2['dFF_iscell']]

for plane in range(planes):
    dff = dFF_all_planes[plane].T
    rewards = pln0['rewards'][0]  # Replace with actual rewards data
    timedFF = pln0['timedFF'][0]  # Replace with actual time data
    forwardvel = pln0['forwardvel'][0]  # Replace with actual velocity data
    # peri rew
    rewdff = []; meanrewdff = []
    for cell in range(dff.shape[1]): # per cell
        normmeanrewdFF, meanrewdFF, normrewdFF, \
        rewdFF = eye.perireward_binned_activity(dff[:,cell], rewards, timedFF, range_val, binsize)
        rewdff.append(rewdFF); meanrewdff.append(meanrewdFF)
    rewdff = np.array(rewdff); meanrewdff=np.array(meanrewdff)
    __, meanvel, _, rewvel = eye.perireward_binned_activity(forwardvel, rewards, timedFF, range_val, binsize)

    fig, axs = plt.subplots(np.ceil(np.sqrt(dff.shape[1])).astype(int), 
                np.ceil(np.sqrt(dff.shape[1])).astype(int),
                sharex=True)
    axs = axs.flatten()
    
    for cll in range(dff.shape[1] + 1):
        if cll < dff.shape[1]:
            ax = axs[cll]
            for rewind in range(rewdff.shape[2]):
                ax.plot(rewdff[cll, :, rewind], color=[.7, .7, .7])  # Plot each trial
            ax.plot(meanrewdff[cll, :], 'k', linewidth=1.5)  # Mean trial plot each cell
            ax.axvline(range_val/binsize, color='slategray', linestyle='--')    
            ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
            ax.set_xticklabels(range(-range_val, range_val+1, 1))
            ax.set_xlabel('Time (s) from Reward')
            if cll == dff.shape[1]:
                for rewind in range(rewvel.shape[1]):
                    ax.plot(rewvel[:, rewind], 'k')  # Plot each trial
                ax.set_ylabel('Velocity (cm/s)')
                ax.axvline(x=allbins.shape[1]//2, color='b', linestyle='--', linewidth=1.5)  # Mark reward
                ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
                ax.set_xticklabels(range(-range_val, range_val+1, 1))
        else:
            ax.remove()

    fig.suptitle(f'All successful trials, plane {plane+1}')

    # Save the figure and add to the PPTX
    img_path = os.path.join(savedst, f'{an}_successful_trials_cell_profiles_peri_reward_plane{plane+1}.svg')
    fig.savefig(img_path, format='svg')
    plt.close(fig)
