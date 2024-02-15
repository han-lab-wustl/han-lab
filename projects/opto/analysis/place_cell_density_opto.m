% calc pre reward enrichment during learning
clear all; close all
% mouse_name = "e216";
mice = ["e216", "e218", "e217", "e201", "e186"];
cond = ["vip", "vip", "vip", "sst", "pv"];%, "pv"];
dys_s = {[7 8 9 37 38 39 40 41 42 44 45 46 48 50:53, 55:59], ...
    [20,21,22,23,35,36,37,38,39,40,41,...
     42,43,44,45,47 48 49 50 51 52 55 56],[2:9], ...
     [52:59], [2:5,31,32,33]};
% experiment conditions: preopto=-1; optoep=3/2; control day1=0; control
% day2=1
opto_eps = {[-1 -1 -1 2 -1 0 1 3 -1 -1 0 1 2 3 0 1 2 0 1 2 0 1],...
    [-1 -1 -1 -1,3 0 1 2 0 1 3,0 1 2, 0 3 0 1 2 0 1 2 0], ...
    [-1 -1 -1 -1 2 3 2 0], ...
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
    load(fullfile(daypth.folder,daypth.name), 'dFF', 'changeRewLoc', 'putative_pcs', 'VR', 'ybinned',...
        'trialnum', 'coms');
    eps = find(changeRewLoc>0);
    eps = [eps length(changeRewLoc)];   
    pcs = reshape(cell2mat(putative_pcs), [length(putative_pcs{1}), length(putative_pcs)]);
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
    eprng = eps(ep):eps(ep+1);
    [pre,post,rew] = get_place_cell_density(eprng, dFF, trialnum, pcs, coms, rewlocs(ep), ybinned, ep);
    density{epind}=[pre,post,rew];
    eprng = eps(ep-1):eps(ep); % normalize to previous epoch
    [pre_,post_,rew_] = get_place_cell_density(eprng, dFF, trialnum, pcs, coms, rewlocs(ep), ybinned, ep);
    density{epind}=[pre-pre_,post-post_,rew-rew_];
    epind = epind+1;
    disp(fullfile(daypth.folder,daypth.name))
end
density_m{m} = density;
end
%%
density_rewopto = {}; density_rewctrl = {};
density_prerewopto = {}; density_prerewctrl = {};
% get density of opto days vs. reg
for m=4:5%length(mice)
    density=density_m{m};
    density_opto_=density(opto_eps{m}>1);
    density_ctrl_=density(opto_eps{m}==-1);
    density_rewopto{m} = cell2mat(cellfun(@(x) x(3), density_opto_, 'UniformOutput',false));
    density_rewctrl{m} = cell2mat(cellfun(@(x) x(3), density_ctrl_, 'UniformOutput',false));
    density_prerewopto{m} = cell2mat(cellfun(@(x) x(1), density_opto_, 'UniformOutput',false));
    density_prerewctrl{m} = cell2mat(cellfun(@(x) x(1), density_ctrl_, 'UniformOutput',false));
end

density_rewopto = cell2mat(density_rewopto);
density_rewctrl = cell2mat(density_rewctrl);
density_prerewopto = cell2mat(density_prerewopto);
density_prerewctrl = cell2mat(density_prerewctrl);

figure; plot(1, density_rewopto, 'ko'); hold on; plot(2, density_rewctrl, 'ko'); xlim([0 3])
[h,p,i,stats] = ttest2(density_rewopto, density_rewctrl)