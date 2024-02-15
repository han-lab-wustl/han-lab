% takes the difference between the cosine similarity b/wn real and shuffled
% maps in opto and ctrl epochs
% numbers look about the same...
% epoch
clear all; close all
% mouse_name = "e216";
mice = ["e216", "e218", "e201", "e186"];
cond = ["vip", "vip", "sst", "pv"];%, "pv"];
dys_s = {[7 8 9 37 38 39 40 41 42 44 45 46 48 50:53, 55:59], ...
    [20,21,22,23,35,36,37,38,39,40,41,...
     42,43,44,45,47 48 49 50 51 52 55 56], ...
     [52:59], [2:5,31,32,33]};
% experiment conditions: preopto=-1; optoep=3/2; control day1=0; control
% day2=1
opto_eps = {[-1 -1 -1 2 -1 0 1 3 -1 -1 0 1 2 3 0 1 2 0 1 2 0 1],...
    [-1 -1 -1 -1,3 0 1 2 0 1 3,0 1 2, 0 3 0 1 2 0 1 2 0], ...
    [-1 -1 -1 2 3 0 2 3],...
    [-1 -1 -1 -1 2 3 2]};
src = ["X:\vipcre", "X:\vipcre", 'Y:\analysis\fmats', ...
    'Y:\analysis\fmats'];
rates_m = {};
for m=1:length(mice)
dys = dys_s{m};
opto_ep = opto_eps{m};
mouse_name = mice{m}; condm = cond{m};
epind = 1; % for indexing
% ntrials = 8; % get licks for last n trials
diff_comparisons = [];
for dy=dys
    if condm=="vip"
        daypth = dir(fullfile(src(m), mouse_name, string(dy), "**\*Fall.mat"));
    else
        daypth = dir(fullfile(src(m), mouse_name, 'days', sprintf('%s_day%03d*.mat', mouse_name, dy))); 
    end
    load(fullfile(daypth.folder,daypth.name));
    eps = find(changeRewLoc>0);
    eps = [eps length(changeRewLoc)];    
    % zahra hard coded to be consistent with the dopamine pipeline
    thres = 5; % 5 cm/s is the velocity filter, only get
    % frames when the animal is moving faster than that
    ftol = 10; % number of frames length minimum to be considered stopped
    ntrials = 8; % e.g. last 8 trials to compare    
    plns = [0]; % number of planes
    % vars to get com and tuning curves
    bin_size = 3; % cm
    Fs = 31.25;
    try
        gainf = 1/VR.scalingFACTOR;
    catch
        gainf = 3/2; % 3/2 VS. 1; in this pipeline the gain is multiplied everywhere
    end
    track_length = 180*gainf;
    [tuning_curves, coms] = make_tuning_curves(eps, changeRewLoc, trialnum, rewards, ybinned, gainf, ntrials,...
    licks, forwardvel, thres, Fs, ftol, bin_size, track_length, stat, iscell, plns, dFF, putative_pcs);

    compep = opto_ep(epind);
    if compep<2 % for control days
        compep=2;
    end
    prevcompep = compep-1;
    [real,shuf] = get_tuning_curve_dist(tuning_curves{prevcompep}', ...
                tuning_curves{compep}');  
    realcs = median(real, 'omitnan'); % average cosine similarity of all cell
    shufcs = median(median(shuf, 'omitnan'));
    diffcs = realcs/shufcs; % some kind of similarity index
    diff_comparisons(epind) = diffcs;    
    epind = epind+1;
    disp(fullfile(daypth.folder,daypth.name))
end
rates_m{m} = diff_comparisons;
end
diff_opto = {}; diff_ctrl = {};
% get rates of opto days
for m=1:2
    rates=rates_m{m};
    rates_opto=rates(opto_eps{m}>1);
    rates_ctrl=rates(opto_eps{m}==-1);
    diff_opto{m} = rates_opto;
    diff_ctrl{m}= rates_ctrl;
end
figure; 
plot(1, cell2mat(diff_opto), 'ko'); hold on; plot(2, cell2mat(diff_ctrl), 'ko'); xlim([0 3])