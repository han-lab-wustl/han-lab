import pickle, matplotlib.pyplot as plt, scipy
import matplotlib as mpl
import numpy as np
mpl.use('TkAgg')
from shapely.geometry import Polygon

def perireward_binned_activity(dFF, rewards, timedFF, range_val, binsize):
    """adaptation of gerardo's code to align IN BOTH TIME AND POSITION, dff or pose data to 
    rewards within a certain window

    Args:
        dFF (_type_): _description_
        rewards (_type_): _description_
        timedFF (_type_): _description_
        range_val (_type_): _description_
        binsize (_type_): _description_

    Returns:
        _type_: _description_
    """
    Rewindx = np.where(rewards)[0]
    rewdFF = np.ones((int(np.ceil(range_val * 2 / binsize)), len(Rewindx)))*np.nan

    for rr in range(0,len(Rewindx)):
        rewtime = timedFF[Rewindx[rr]]
        currentrewchecks = np.where((timedFF > rewtime - range_val) & (timedFF <= rewtime + range_val))[0]
        currentrewcheckscell = consecutive_stretch(currentrewchecks) # gets consecutive stretch of reward ind
        # check for missing vals
        currentrewcheckscell = [xx for xx in currentrewcheckscell if len(xx)>0]
        currentrewcheckscell = np.array(currentrewcheckscell) # reformat for py
        currentrewardlogical = np.array([sum(Rewindx[rr]==x).astype(bool) for x in currentrewcheckscell])
        val = 0
        for bin_val in range(int(np.ceil(range_val * 2 / binsize))):
            val = bin_val+1
            currentidxt = np.where((timedFF>(rewtime - range_val + (val * binsize) - binsize)) & (timedFF <= rewtime - range_val + val * binsize))[0]
            checks = consecutive_stretch(currentidxt)
            checks = [list(xx) for xx in checks]
            if len(checks[0]) > 0:
                currentidxlogical = np.array([np.isin(x, currentrewcheckscell[currentrewardlogical][0]) \
                                for x in checks])
                for i,cidx in enumerate(currentidxlogical):
                    cidx = [bool(xx) for xx in cidx]
                    if sum(cidx)>0:
                        checkidx = np.array(np.array(checks)[i])[np.array(cidx)]
                        rewdFF[bin_val, rr] = np.nanmean(dFF[checkidx])

    meanrewdFF = np.nanmean(rewdFF, axis=1)    
    # allbins = np.array([round(-range_val + bin_val * binsize - binsize, 13) for bin_val in range(int(np.ceil(range_val * 2 / binsize)))])
    normmeanrewdFF = (meanrewdFF-np.min(meanrewdFF)) / (np.max(meanrewdFF) - np.min(meanrewdFF))
    normrewdFF = np.array([(xx-np.min(xx))/((np.max(xx)-np.min(xx))) for xx in rewdFF.T])
    return normmeanrewdFF, meanrewdFF, normrewdFF, rewdFF
################################FUNCTION DEFINITIONS################################

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

## cell 

vralign = r"Y:\DLC\dlc_mixedmodel2\E201_02_May_2023_vr_dlc_align.p"
with open(vralign, "rb") as fp: #unpickle
    vralign = pickle.load(fp)

# plt.figure; plt.plot(area)
threshold = 1e-3
vralign['TongueTop_x'][vralign['TongueTop_likelihood'].astype('float32') < threshold] = 0
vralign['TongueTop_y'][vralign['TongueTop_likelihood'].astype('float32') < threshold] = 0
vralign['TongueTip_x'][vralign['TongueTip_likelihood'].astype('float32') < threshold] = 0
vralign['TongueTip_y'][vralign['TongueTip_likelihood'].astype('float32') < threshold] = 0
vralign['TongueBottom_x'][vralign['TongueBottom_likelihood'].astype('float32') < threshold] = 0
vralign['TongueBottom_y'][vralign['TongueBottom_likelihood'].astype('float32') < threshold] = 0
y = np.array([vralign['TongueTop_y'], vralign['TongueTip_y'],vralign['TongueBottom_y']])
x = np.array([vralign['TongueTop_x'], vralign['TongueTip_x'],vralign['TongueBottom_x']])
x_ = consecutive_stretch(np.where(x>0)[1])
y_ = consecutive_stretch(np.where(y>0)[1])
x_ = [min(xx) for xx in x_ if len(xx)>0]
y_ = [min(xx) for xx in y_ if len(xx)>0]
xm = x; ym=y
xm[:,~np.array(x_)]=0
ym[:,~np.array(y_)]=0
rew = np.hstack(vralign['rewards'])#[x_]
lick = np.hstack(vralign['licks'])#[x_]
lick = lick>0
lick_fix = consecutive_stretch(np.where(lick>0)[0])
lick_mask = np.zeros(lick.shape)
lick_ind = [min(xx) for xx in lick_fix if len(xx)>0]
lick_mask[lick_ind] = 1

areas = []
for i in range(xm.shape[1]):
    pgon = Polygon(zip(xm[:,i], ym[:,i])) # Assuming the OP's x,y coordinates
    areas.append(pgon.area)
areas = np.array(areas)
areas[areas>2000]=0
areas = scipy.ndimage.gaussian_filter(areas,2)


%matplotlib inline
# plt.figure; plt.hist(vralign['TongueTip_likelihood'])
r = np.random.randint(1000, len(areas))
plt.figure; 
# plt.plot(vralign['TongueTip_y'][r:r+1500])
# plt.plot(vralign['TongueBottom_y'][r:r+1500])
plt.plot(((lick_mask)*1000)[r:r+500]) 
plt.plot(areas[r:r+500])
plt.plot(((rew)*1500)[r:r+500], 'k')
plt.ylim([0, 2000])


normmeanrewdFF, meanrewdFF, normrewdFF, rewdFF = perireward_binned_activity(areas, rew==1, \
            np.hstack(vralign['timedFF']), 8, 0.2)
_, meanrewlick, __, rewlick = perireward_binned_activity(lick_mask, rew==1, \
            np.hstack(vralign['timedFF']), 8, 0.2)
plt.figure; plt.imshow(rewdFF.T/rewlick.T, cmap = 'Reds')
plt.figure; plt.imshow(rewlick.T)
fig, axes = plt.subplots(nrows=2,ncols=1)
axes[0].plot(meanrewdFF) 
# axes[1].plot(meanrewlick) 
# ax.plot(meanrewlick)