% get place cell fraction near reward
clear all
close all
load('X:\vipcre\e218\50\231217_ZD_000_000\suite2p\plane0\Fall.mat');
optoep = 2; gainf = 1.5;
% preprocessing
eps = find(changeRewLoc>0);
eps = [eps length(changeRewLoc)];
track_length = 180*gainf;
bin_size=3;
nbins = track_length/bin_size;
ybinned = ybinned*gainf;
rewlocs = changeRewLoc(changeRewLoc>0)*gainf;
thres = 5; % 5 cm/s is the velocity filter, only get
ftol = 10; % number of frames length minimum to be considered stopped
ntrials = 5; % e.g. last 8 trials to compare    
plns = [0]; % number of planes
Fs = 31.25;
%%
optoep = 2;
[tuning_curves, coms] = make_tuning_curves(eps, changeRewLoc, trialnum, rewards, ybinned, gainf, ntrials,...
    licks, forwardvel, thres, Fs, ftol, bin_size, track_length, stat, iscell, plns, Fc3, putative_pcs);

com_opto = coms{optoep};
% get # of coms near reward zone
rad = 30;
rewloc = rewlocs(optoep);
com_rewloc = sum(com_opto>rewloc-rad & com_opto<rewloc+rad, 'omitnan');
area_rewloc = (rad*2)/track_length;
frac_pc = com_rewloc/length(com_opto>30)
