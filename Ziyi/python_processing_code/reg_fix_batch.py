import os
import re
import numpy as np
from tkinter import filedialog, Tk

def pick_directories():
    root = Tk()
    root.withdraw()  # We don't want a full GUI, so keep the root window from appearing
    directories = []
    while True:
        directory = filedialog.askdirectory(title="Select a directory, or Cancel to finish")
        if directory:
            directories.append(directory)
        else:
            break
    return directories

# Use the GUI to pick directories
root_directories = pick_directories()

planes = [0, 1, 2, 3]  # Specify number of planes
for srcpth in root_directories:
    pths = [os.path.join(srcpth, 'suite2p', f'plane{plane}', 'reg_tif') for plane in planes]
    
    for pth in pths:
        fls = [os.path.join(pth, xx) for xx in os.listdir(pth) if 'tif' in xx]
        order = np.array([int(re.findall(r'\d+', os.path.basename(xx))[0]) for xx in fls])

        for i, fl in enumerate(fls):
            new_name = os.path.join(pth, f'file{order[i]:06d}.tif')
            os.rename(fl, new_name)
            print(new_name)  # Uncomment to print the new file paths
