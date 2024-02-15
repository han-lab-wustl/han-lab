# Zahra
# make csv of video and file num
# use to check whether tif folder can be converted to video

import os, pandas as pd

<<<<<<< HEAD
src = r"F:\240129-240204"
=======
<<<<<<< HEAD
src = r"F:\raw_tail_vids\2"
=======
src = r"E:\tail_temp\240115-240121"
>>>>>>> 097c18b96088986bd88fb2454ae335227b13e6c2
>>>>>>> 03257286f1285e1a4af2ef484af53f0470f37fb3
vids = [os.path.join(src, xx) for xx in os.listdir(src) if 'csv' not in xx]
fls = [len(os.listdir(xx)) for xx in vids]

df = pd.DataFrame(vids, fls)
# if not os.path.exists(os.path.join(src, 'dlc_curation')): os.mkdir(os.path.join(src, 'dlc_curation'))
# dst = os.path.join(src, 'dlc_curation', 'to_convert')
# if not os.path.exists(dst): os.mkdir(dst)
df.to_csv(os.path.join(src, 'video_list.csv'))