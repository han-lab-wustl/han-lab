% calc pre reward enrichment during learning
clear all; close all
% mouse_name = "e216";
mice = ["e216", "e218", "e217", "e201", "e186"];
cond = ["vip", "vip", "vip", "sst", "pv"];%, "pv"];
dys_s = {[7 8 9 37 38 39 40 41 42 44 45 46 48 50:53, 55:59], ...
    [20,21,22,23,35,36,37,38,39,40,41,...
     42,43,44,45,47 48 49 50 51 52 55 56],[2:9 12], ...
     [52:59], [2:5,31,32,33]};
Fs = 31.25; % assumes 1 plane imaging
% experiment conditions: preopto=-1; optoep=3/2; control day1=0; control
% day2=1
opto_eps = {[-1 -1 -1 2 -1 0 1 3 -1 -1 0 1 2 3 0 1 2 0 1 2 0 1],...
    [-1 -1 -1 -1,3 0 1 2 0 1 3,0 1 2, 0 3 0 1 2 0 1 2 0], ...
    [-1 -1 -1 -1 2 3 2 0 2], ...
    [-1 -1 -1 2 3 0 2 3],...
    [-1 -1 -1 -1 2 3 2]};
src = ["X:\vipcre", "X:\vipcre", "X:\vipcre", 'Y:\analysis\fmats', ...
    'Y:\analysis\fmats'];
density_m = {};
for m=1:length(mice)
dys = dys_s{m};
opto_ep = opto_eps{m};
mouse_name = mice{m}; condm = cond{m};
epind = 1; % for indexing
% ntrials = 8; % get licks for last n trials
density = {};
for dy=dys
    if condm=="vip"
        daypth = dir(fullfile(src(m), mouse_name, string(dy), "**\*Fall.mat"));
    else
        daypth = dir(fullfile(src(m), mouse_name, 'days', sprintf('%s_day%03d*.mat', mouse_name, dy))); 
    end
    load(fullfile(daypth.folder,daypth.name), 'Fc3', 'changeRewLoc', 'putative_pcs', 'VR', 'ybinned',...
        'trialnum', 'forwardvel');
    fv = forwardvel;
    eps = find(changeRewLoc>0);
    eps = [eps length(changeRewLoc)];   
    pcs = reshape(cell2mat(putative_pcs), [length(putative_pcs{1}), length(putative_pcs)]);
    % zahra hard coded to be consistent with the dopamine pipeline
    thres = 5; % 5 cm/s is the velocity filter, only get
    % frames when the animal is moving faster than that
    ftol = 10; % number of frames length minimum to be considered stopped    
    try
        gainf = 1/VR.scalingFACTOR;
    catch
        gainf = 3/2; % 3/2 VS. 1; in this pipeline the gain is multiplied everywhere
    end
    ep = opto_ep(epind); % pick ep to analyze
    if ep<2
        ep=2;
    end
    track_length = 180*gainf; ybinned = ybinned*gainf; rewlocs = changeRewLoc(changeRewLoc>0)*(gainf); 
    window = 20; % cm 
    bin_size = 3; % cm
    nBins = track_length/bin_size; % 270/3 cm bins typically
    eprng = eps(ep):eps(ep+1);
    [pre,post,rew] = get_place_cell_density(eprng, Fc3, fv, trialnum, pcs, track_length, rewlocs(ep), ybinned, Fs, ...
    window, nBins, thres, ftol);
    eprng = eps(ep-1):eps(ep); % normalize to previous epoch
    [pre_,post_,rew_] = get_place_cell_density(eprng, Fc3, fv, trialnum, pcs, track_length, rewlocs(ep), ybinned, Fs, ...
    window, nBins, thres, ftol);
    density{epind}={pre,post,rew};
    % density{epind}=[pre-pre_,post-post_,rew-rew_];
    epind = epind+1;
    disp(fullfile(daypth.folder,daypth.name))
end
density_m{m} = density;
end
%%
density_rewopto = {}; density_rewctrl = {};
density_prerewopto = {}; density_prerewctrl = {}; density_postrewopto = {}; density_postrewctrl={};
% get density of opto days vs. reg
for m=1:3%length(mice)
    density=density_m{m};
    density_opto_=density(opto_eps{m}>1);
    density_ctrl_=density(opto_eps{m}<1);
    density_rewopto{m} = cell2mat(cellfun(@(x) x{3}, density_opto_, 'UniformOutput',false));
    density_rewctrl{m} = cell2mat(cellfun(@(x) x{3}, density_ctrl_, 'UniformOutput',false));
    density_postrewopto{m} = cell2mat(cellfun(@(x) x{2}, density_opto_, 'UniformOutput',false));
    density_postrewctrl{m} = cell2mat(cellfun(@(x) x{2}, density_ctrl_, 'UniformOutput',false));
    density_prerewopto{m} = cell2mat(cellfun(@(x) x{1}, density_opto_, 'UniformOutput',false));
    density_prerewctrl{m} = cell2mat(cellfun(@(x) x{1}, density_ctrl_, 'UniformOutput',false));
end

density_rewopto = cell2mat(density_rewopto);
density_rewctrl = cell2mat(density_rewctrl);
density_postrewopto = cell2mat(density_postrewopto);
density_postrewctrl = cell2mat(density_postrewctrl);
density_prerewopto = cell2mat(density_prerewopto);
density_prerewctrl = cell2mat(density_prerewctrl);

y = [density_prerewctrl density_rewctrl density_postrewctrl ...
    density_prerewopto density_rewopto density_postrewopto];
y_ = {density_prerewctrl, density_rewctrl, density_postrewctrl, ...
    density_prerewopto, density_rewopto, density_postrewopto};
x = [repelem(1,length(density_prerewctrl)) repelem(2,length(density_rewctrl)) repelem(3,length(density_postrewctrl))...
    repelem(4,length(density_prerewopto)) repelem(5,length(density_rewopto)) repelem(6,length(density_postrewopto))];
figure;
means = cellfun(@mean, y_);
bar(means, 'FaceColor', 'w'); hold on
swarmchart(x,y,'ko')
ylabel('Sparsity Index (last trials-first trials)')
yerr = {density_prerewctrl density_rewctrl density_postrewctrl ...
    density_prerewopto density_rewopto density_postrewopto};
err = [];
for i=1:length(yerr)
    err(i) =(std(yerr{i},'omitnan')/sqrt(size(yerr{i},2))); 
end
er = errorbar([1:6],means,err);
er.Color = [0 0 0];                            
er.LineStyle = 'none';  
xticklabels(["Control Pre-Reward", "Control Reward Location", "Control Post-Reward", "VIP stGtACR Pre-Reward", ...
    "VIP stGtACR Reward Location", "VIP stGtACR Post-Reward"])

[h,p,i,stats] = ttest2(density_prerewopto, density_prerewctrl)