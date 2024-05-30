import numpy as np, sys, os, scipy.io as sio, h5py, scipy
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.DLC_behavior_classification import eye
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
plt.rc('font', size=16)          # controls default text sizes

savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\thesis_proposal'

# Load MATLAB data
matfl = r"Z:\vip_dff_probes_3probe.mat"
# matrix contents: dy,pln,ep
# for each arr: probes, probes-ctrl, success
mat = sio.loadmat(matfl)
#%%
dffprobes = mat['dff_probes']
dys = range(5)
rs_dy = []; rs_ctrl_dy = []
for dy in dys:
    # just day 3 for now
    dffprobes_dy = dffprobes[dy,:,:]
    # probes
    fig,axes = plt.subplots(nrows=2,ncols=1,sharex=True)
    for ep in range(1,dffprobes_dy.shape[1]-1):
        dff = dffprobes_dy[:,ep]
        dff = np.concatenate(dff)
        ax = axes[ep-1]
        ax.imshow(dff[:,:int(dff.shape[1]/3)])
        ax.set_title('Probes')
    fig.suptitle(f'Day {dy}, all cells')
    # correct trials
    fig,axes = plt.subplots(nrows=2,ncols=1,sharex=True)
    for ep in range(1,dffprobes_dy.shape[1]-1):
        dff = dffprobes_dy[:,ep]
        dff = np.concatenate(dff)
        ax = axes[ep-1]
        ax.imshow(dff[:,int(dff.shape[1]/3)*2:int(dff.shape[1]/3)*3])
        ax.set_title('Correct Trials')
    fig.suptitle(f'Day {dy}, all cells')
    rs = []; rs_c = []
    for cll in range(dff.shape[0]):
        r,pval = scipy.stats.pearsonr(dff[cll,:int(dff.shape[1]/3)],
            dff[cll,int(dff.shape[1]/3)*2:int(dff.shape[1]/3)*3])
        rc,pval = scipy.stats.pearsonr(dff[cll,int(dff.shape[1]/3):int(dff.shape[1]/3)*2],
            dff[cll,int(dff.shape[1]/3)*2:int(dff.shape[1]/3)*3])
        rs.append(r); rs_c.append(rc)
    rs_dy.append(rs); rs_ctrl_dy.append(rs_c)
#%%
def KL(P,Q):
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0. """
    epsilon = 0.00001

    # You may want to instead make copies to avoid changing the np arrays.
    P = P+epsilon
    Q = Q+epsilon

    divergence = np.sum(P*np.log(P/Q))
    return divergence

fig, ax = plt.subplots()
ax.hist(np.concatenate(rs_dy), alpha=0.5, color = 'k', label='Probes')
ax.hist(np.concatenate(rs_ctrl_dy),alpha=0.5, color = 'darkslategray',label='Probes to Shuffled Loc.')
ax.legend()
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('# Cells')
ax.set_xlabel('Pearson r')
scipy.stats.ttest_ind(np.concatenate(rs_dy),np.concatenate(rs_ctrl_dy))
# %%
