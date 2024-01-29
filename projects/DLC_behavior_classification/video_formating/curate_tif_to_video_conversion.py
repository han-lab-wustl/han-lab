# Zahra
# make csv of video and file num
# use to check whether tif folder can be converted to video

import os, pandas as pd

src = r"H:\tailvideos"
vids = [os.path.join(src, xx) for xx in os.listdir(src) if 'csv' not in xx]
fls = [len(os.listdir(xx)) for xx in vids]

df = pd.DataFrame(vids, fls)
# if not os.path.exists(os.path.join(src, 'dlc_curation')): os.mkdir(os.path.join(src, 'dlc_curation'))
# dst = os.path.join(src, 'dlc_curation', 'to_convert')
# if not os.path.exists(dst): os.mkdir(dst)
df.to_csv(os.path.join(src, 'video_list.csv'))