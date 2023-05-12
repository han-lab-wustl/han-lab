import os, sys
from utils.utils import listdir

src = r'Y:\sstcre_imaging\e200'
days = listdir(src)
pth = [listdir(xx) for xx in days]