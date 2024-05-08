import os
import re
import numpy as np
import pandas as pd

# Read the Excel file
excel_path = 'E:\Ziyi\Excel_Worksheet.xlsx'
df = pd.read_excel(excel_path)
first_column = df.iloc[:, 0]
planes = [0,1,2,3] # specify number of planes

# Iterate through each row in the DataFrame
for srcpth in first_column:
    
    pths = []
    for plane in planes:
        pths.append(os.path.join(srcpth, 'suite2p', f'plane{plane}', 'reg_tif'))
    
    for pth in pths:
        fls = [os.path.join(pth, xx) for xx in os.listdir(pth) if 'tif' in xx]
        order = np.array([int(re.findall(r'\d+', os.path.basename(xx))[0]) for xx in fls])
        
        for i, fl in enumerate(fls):
            os.rename(fl, os.path.join(pth, f'file{order[i]:06d}.tif'))
