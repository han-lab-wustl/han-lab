# Zahra
# find videos that match vr files with a specific behavior
# used to pool animals for dlc analysis

import pandas as pd, os, numpy as np, sys
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom your clone
from datetime import datetime
from utils.utils import listdir

src = r'\\storage1.ris.wustl.edu\ebhan\Active\DopamineData'
vrfld = r'\\storage1.ris.wustl.edu\ebhan\Active\all_vr_data'
vidfld = r'\\storage1.ris.wustl.edu\ebhan\Active\mouse_videos\new_eye_videos'

mouse = 'e232'
svpth = rf'C:\Users\Han\Desktop\{mouse}_eye_video_lut.csv'
structs = os.path.join(src, mouse+'_hrz')
days = [xx for xx in os.listdir(structs) if '.mat' not in xx]
df = pd.DataFrame()
df['day'] = days
df['mouse'] = [mouse]*len(df)
for day in days:
    pth = os.path.join(structs,day)
    print(pth)
    imgfld = [xx for xx in os.listdir(pth) if '00' in xx][0]
    # Extract the date part
    date_part = imgfld[:6]
    # Convert to datetime object
    date_object = datetime.strptime(date_part, '%y%m%d')
    # Convert to desired format with month name
    formatted_date = date_object.strftime('%d_%b_%Y')
    an_upper = mouse.upper()
    vrstr = an_upper+'_'+formatted_date
    vrfl = [xx for xx in os.listdir(vrfld) if vrstr in xx]
    if len(vrfl)>0:
        vrfl=vrfl[0]
        vrflpth = os.path.join(vrfld, vrfl) # vr file
        vidstr = date_object.strftime('%y%m%d')+'_'+an_upper
        avifl = [xx for xx in os.listdir(vidfld) if vidstr in xx]
        if len(avifl)>1:
            avifl = [xx for xx in avifl if '2' in xx]    
        elif len(avifl)==0:
            avifl = 'no_video_detected'
        else:
            avifl=avifl[0]
        avipth = os.path.join(vidfld, avifl) # avi file
        df.loc[((df.day==day) & (df.mouse==mouse), 'date')] = formatted_date
        df.loc[((df.day==day) & (df.mouse==mouse), 'eye_video_path')] = avipth
        df.loc[((df.day==day) & (df.mouse==mouse), 'vr_file_path')] = vrflpth
    
df.to_csv(svpth)