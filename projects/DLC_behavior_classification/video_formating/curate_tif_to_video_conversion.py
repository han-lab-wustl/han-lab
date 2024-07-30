# Zahra
# make csv of video and file num
# use to check whether tif folder can be converted to video

import os, pandas as pd

src = r'E:'
for weekfld in os.listdir(src):
    if '24' in weekfld:
        weekfld = os.path.join(src,weekfld)
        print(weekfld)
        vids = [os.path.join(weekfld, xx) for xx in os.listdir(weekfld) if 'csv' not in xx]
        fls = [len(os.listdir(xx)) for xx in vids]

        df = pd.DataFrame(vids, fls)
        # if not os.path.exists(os.path.join(src, 'dlc_curation')): os.mkdir(os.path.join(src, 'dlc_curation'))
        # dst = os.path.join(src, 'dlc_curation', 'to_convert')
        # if not os.path.exists(dst): os.mkdir(dst)
        df.to_csv(os.path.join(weekfld, 'video_list.csv'))