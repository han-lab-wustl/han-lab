import os, numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
sys.path.append(r'C:\Users\HanLab\Documents\GitHub\han-lab') ## custom to your clone
from projects.DLC_behavior_classification import eye
from pathlib import Path
import matplotlib.backends.backend_pdf
import matplotlib, seaborn as sns
from projects.memory.behavior import get_success_failure_trials, consecutive_stretch
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
import matplotlib.patches as patches


srcs = [r"E:\Ziyi\Data\250103_ZH\250103_ZH_000_000"]
planelut = {0: 'SLM', 1: 'SR', 2: 'SP', 3: 'SO'}
range_val = 8
binsize = 0.2

# Create a single figure and multiple subplots
fig, axes = plt.subplots(4, 1, figsize=(10, 20))  # Adjust the figure size as needed

for idx, src in enumerate(srcs):
    print(src)
    stimspth = list(Path(src).rglob('*000*.mat'))[0]
    stims = scipy.io.loadmat(stimspth)
    stims = np.hstack(stims['stims'])

    for path in Path(src).rglob('params.mat'):
        params = scipy.io.loadmat(path)
        VR = params['VR'][0][0]; gainf = VR[14][0][0]
        planenum = os.path.basename(os.path.dirname(os.path.dirname(path)))
        pln = int(planenum[-1])
        dff = np.hstack(params['params'][0][0][6][0][0]) / np.nanmean(np.hstack(params['params'][0][0][6][0][0]))
        timedFF = np.hstack(params['timedFF'])
        dffdf = pd.DataFrame({'dff': dff})
        dff = np.hstack(dffdf.rolling(3).mean().values)
        offpln = pln + 1 if pln < 3 else pln - 1
        startofstims = consecutive_stretch(np.where(stims[offpln::4])[0])
        min_iind = [min(xx) for xx in startofstims if len(xx) > 0]
        startofstims = np.zeros_like(dff)
        startofstims[min_iind] = 1

        ax = axes[pln]  # Choose subplot
        normmeanrewdFF, meanrewdFF, normrewdFF, rewdFF = eye.perireward_binned_activity(dff, startofstims, timedFF, range_val, binsize)
        ax.plot(meanrewdFF, color='slategray')
        ax.fill_between(range(0, int(range_val/binsize) * 2),
                        meanrewdFF - scipy.stats.sem(rewdFF, axis=1, nan_policy='omit'),
                        meanrewdFF + scipy.stats.sem(rewdFF, axis=1, nan_policy='omit'),
                        color='slategray', alpha=0.4)
        ax.set_xticks(range(0, (int(range_val/binsize) * 2) + 1, 5))
        ax.set_xticklabels(range(-range_val, range_val + 1, 1))
        ax.set_title(f'Peri-stim, 200mA, plane {pln} \n {src}')

# Tight layout to ensure no overlap
plt.tight_layout()

# Save the figure to a PDF file
#pdf_path = r"C:\path\to\save\figure.pdf"
#plt.savefig(pdf_path)
plt.show()
