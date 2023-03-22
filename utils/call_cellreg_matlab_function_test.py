import matlab.engine
eng = matlab.engine.start_matlab()

src = 'Z:\sstcre_imaging'
mouse = "e201"
day = "1"
imgfl = [os.path.join(src, mouse, day, xx) for xx in os.listdir(os.path.join(src, mouse, day)) if "000" in xx]
pth = 'Z:\sstcre_analysis\Fall.mat'
eng.format_conversion_Suite2p_CNMF_e(pth,nargout=0)

fmatfl='Z:/sstcre_analysis/fmats/e200_week1/e200_day001_Fall.mat'
vrfl='Z:/sstcre_imaging\e200/3/behavior/vr\E200_16_Feb_2023_time(07_26_44).mat'
savepth = eng.VRalign(fmatfl,vrfl,nargout=1)