import numpy as np
from scipy.io import loadmat, savemat
import os
from tkinter import filedialog
from tifffile import imsave
import imageio
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import sbxreader

def exp2_func(x, a, b, c, d):
    return a * np.exp(b * x) + c * np.exp(d * x)

def make_tifs_wo_opto_artifact(filepath=None, filename=None, dir_type='bi'):
    if filepath is None or filename is None:
        root = os.path.dirname(os.path.abspath(__file__))
        file_path = filedialog.askopenfilename(initialdir=root, title="Choose SBX file", filetypes=[("SBX files", "*.sbx")])
        filepath, filename = os.path.split(file_path)
    elif filepath is not None and filename is None:
        raise ValueError("If filepath is provided, filename must also be provided.")

    os.chdir(filepath)
    z = sbxreader.sbx_memmap(stripped_filename, 1, 1)
    info = z.info

    numframes = info['max_idx'] + 1

    lenVid = 3000
    stims = []

    for ii in range(1, int(np.ceil(numframes / lenVid)) + 1):
        if ii > 9:
            currfile = f"{stripped_filename}_x{ii}.mat"
        else:
            currfile = f"{stripped_filename}_{ii}.mat"

        if dir_type == 'uni' or dir_type == 'uni new':
            chtemp = sbxread(stripped_filename, (ii - 1) * lenVid, min(lenVid, (numframes - (ii - 1) * lenVid)))
            chtemp = np.double(np.squeeze(chtemp))
            chtemp = chtemp[:, 45:730, :]
        elif dir_type == 'bi':
            chtemp = sbxread(stripped_filename, (ii - 1) * lenVid, min(lenVid, (numframes - (ii - 1) * lenVid)))
            chtemp = np.double(np.squeeze(chtemp))
            chtemp = chtemp[:, 110:721, :]
        elif dir_type == 'bi new':
            chtemp = sbxread(stripped_filename, (ii - 1) * lenVid, min(lenVid, (numframes - (ii - 1) * lenVid)))
            chtemp = np.double(np.squeeze(chtemp))
            for currimage in range(chtemp.shape[2]):
                imag = chtemp[:, :, currimage]
                startpoint = (80 * imag.shape[1]) + 1
                startline = 80
                rightborder = 740
                leftborder = 76
                imag[:, leftborder:rightborder] = np.nan
                dummy = []
                for y in range(imag.shape[0]):
                    if y % 2 == 1:
                        dummy.extend(imag[y, :])
                    else:
                        dummy.extend(np.flipud(imag[y, :]))

                dummy = np.array(dummy)
                dummy[dummy > 65500] = np.nan
                dummy[dummy == 0] = np.nan
                temp = dummy[startpoint:]
                xs = np.where(~np.isnan(temp))[0]

                modl, _ = curve_fit(exp2_func, xs, temp[xs], p0=[1, -1, 1, -1])
                fil = exp2_func(np.arange(len(temp)), *modl)

                filimage = np.full_like(imag, np.nan)
                filimage[:startline, :] = np.tile(np.nanmean(imag[:startline, rightborder+1:], axis=1, keepdims=True), (1, imag.shape[1]))
                for y in range(imag.shape[0] - startline):
                    if y % 2 == 1:
                        filimage[y+startline, :] = fil[y*imag.shape[1]:(y+1)*imag.shape[1]]
                    else:
                        filimage[y+startline, :] = np.flipud(fil[y*imag.shape[1]:(y+1)*imag.shape[1]])

                chtemp[:, :, currimage] -= filimage
            chtemp = chtemp[:, 90:718, :]

        chtemp = ((np.double(chtemp) / 2) - 1)
        chtemp = np.uint16(chtemp)
        temp = np.squeeze(np.nanmean(np.nanmean(chtemp[:20, :, :], axis=0), axis=0))
        tempstims = np.zeros(len(temp), dtype=int)

        for p in range(info['etl_table'].shape[0]):
            currx = np.arange(p, len(temp), info['etl_table'].shape[0])
            temp2 = np.abs(temp[currx] / np.nanmean(temp[currx]) - 1)
            s = find_peaks(temp2, height=0.2)[0]
            tempstims[currx[s]] = 1

        if ii == 1:
            tempstims[:10] = 0

        stims.append(tempstims)

        imageJ_savefilename = os.path.join(filepath, f"{currfile[:-4]}.tif")
        imsave(imageJ_savefilename, chtemp.astype(np.uint16))

    stims = np.concatenate(stims)
    savemat(os.path.join(filepath, f"{stripped_filename}.mat"), {'stims': stims})

if __name__ == "__main__":
    src = r"Z:\chr2_grabda\e232\22\240318_ZD_000_001\240318_ZD_000_001.sbx"
    make_tifs_wo_opto_artifact(filepath=os.path.dirname(src), filename=os.path.basename(src), dir_type='bi')