
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter
import scipy

def load_3d_tiff_stack(file_path):
    # Load the 3D TIFF stack from the specified file path
    stack = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
    return stack

def main(file_path, output_folder='heatmap_stack'):
    stack = load_3d_tiff_stack(file_path)[:100]
    diffs = np.diff(stack, axis=0)
    smoothed_diffs = gaussian_filter(diffs, sigma=1)

    # Compute p-values using a paired t-test between each frame and a reference
    # Here we compare each pair of consecutive frames
    p_values = np.empty(smoothed_diffs.shape[1:])  # shape (z, y, x)

    for y in range(smoothed_diffs.shape[1]):
        for x in range(smoothed_diffs.shape[2]):
            p_values[y, x] = scipy.stats.ttest_rel(stack[:-1, y, x], stack[1:, y, x])[1]


# Example usage
file_path = r'Y:\halo_grabda\e242\26\241207_ZD_000_000\suite2p\plane3\reg_tif\file005000.tif'
output_folder=r'C:\Users\Han\Desktop\so'
p_values = main(file_path,output_folder)

# For visualization, we might want to take the mean across the z-axis (flattening)
mean_p_values = np.mean(p_values, axis=0)

plt.imshow(mean_p_values, cmap='hot', interpolation='nearest')
plt.colorbar(label='P-value')
plt.title('Heatmap of Change in Pixel Values (P-values)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

#%% 
# import behavior
path = r'Y:\halo_grabda\e242\26\241207_ZD_000_000\suite2p\plane3\reg_tif\params.mat'
params = scipy.io.loadmat(path)
VR = params['VR'][0][0][()]
gainf = VR['scalingFACTOR'][0][0]
try:
    rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
except:
    rewsize = 10

planenum = os.path.basename(os.path.dirname(os.path.dirname(path)))
pln = int(planenum[-1])
layer = planelut[pln]
params_keys = params.keys()
keys = params['params'].dtype
# dff is in row 7 - roibasemean3/basemean
dff = np.hstack(params['params'][0][0][6][0][0])/np.nanmean(np.hstack(params['params'][0][0][6][0][0]))

dffdf = pd.DataFrame({'dff': dff})
dff = np.hstack(dffdf.rolling(5).mean().values)
rewards = np.hstack(params['solenoid2'])
velocity = np.hstack(params['forwardvel'])
veldf = pd.DataFrame({'velocity': velocity})
velocity = np.hstack(veldf.rolling(5).mean().values)
trialnum = np.hstack(params['trialnum'])
ybinned = np.hstack(params['ybinned'])/(2/3)
licks = np.hstack(params['licks'])
changeRewLoc = np.hstack(params['changeRewLoc'])
eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/gainf
eps = np.append(eps, len(changeRewLoc))

# plot pre-first reward dop activity  
timedFF = np.hstack(params['timedFF'])
