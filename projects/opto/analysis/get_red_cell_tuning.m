
clear all; close all
% mouse_name = "e218";
% days = [34,35,36,37,38,39,40,41,44,47,50];
% cells_to_plot = {[141, 17,20,7]+1, [453,63,26,38]+1, [111,41,65,2]+1, [72,41,27,14]+1,...
%     [301 17 13 320]+1, [98 33 17 3]+1, [92 20 17 26]+1, [17, 23, 36, 10]+1, [6, 114, 11, 24]+1,...
%      [49 47 6 37]+1, [434,19,77,5]+1}; % indices of red cells from suite2p per day
mice = {"e216", "e217", "e218"};
% mice = {"e186"};%{"e201"};
% dys_s = {[2:5,31,32,33,36, 38 40 41]};%{[52:65]};
dys_s = {[37 41 57 60],[14, 26, 27],...
    [35,38,41,44,47,50]};
opto_ep_s = {[2 3 2 3],[2,3,3],...
    [3 2 3 2 3 2]};
cells_to_plot_s = {{135+1,1655+1,780,2356+1},{16+1,6+1,9+1} ...
    {[453,63,26,38]+1,...
    [301 17 13 320]+1, [17, 23, 36, 10]+1, [6, 114, 11, 24]+1,...
    [49 47 6 37]+1, [434,19,77,5]+1}}; % indices of red cells from suite2p per day
src = "X:\vipcre";
dffs_cp_dys = {};
mind = 1;
for m=1:length(mice)
    mouse_name = mice{m};
    days = dys_s{m};
    cells_to_plot = cells_to_plot_s{m};
    opto_ep = opto_ep_s{m};
    dyind = 1;
for dy=days
    daypth = dir(fullfile(src, mouse_name, string(dy), "**\*Fall.mat"));
    load(fullfile(daypth.folder,daypth.name), 'dFF', 'changeRewLoc', 'VR', 'ybinned', 'forwardvel', ...
        'timedFF', 'rewards', 'licks', 'trialnum');
    disp(fullfile(daypth.folder,daypth.name))
    %%
    grayColor = [.7 .7 .7];
    % get red cells
    eps = find(changeRewLoc>0);
    eps = [eps length(changeRewLoc)];
    gainf = 1/VR.scalingFACTOR;
    rewloc = changeRewLoc(changeRewLoc>0)*gainf;
    rewsize = VR.settings.rewardZone*gainf;
    ypos = ybinned*(gainf);
    velocity = forwardvel;
    dffs_cp = {};
    indtemp = 1;
    for cp=cells_to_plot{dyind}              
        % compare to prev epoch
        rngopto = eps(opto_ep(dyind)):eps(opto_ep(dyind)+1);
        rngpreopto = eps(opto_ep(dyind)-1):eps(opto_ep(dyind));        
        yposopto = ypos(rngopto);
        ypospreopto = ypos(rngpreopto);
        yposoptomask = yposopto<rewloc(opto_ep(dyind))-rewsize; % get rng pre reward
        ypospreoptomask = ypospreopto<rewloc(opto_ep(dyind)-1)-rewsize; % get rng pre reward
        trialoptomask = trialnum(rngopto)>12;
        trialpreoptomask = trialnum(rngpreopto)>12;
        % yposoptomask = ones(1, length(yposopto));
        % ypospreoptomask = ones(1, length(ypospreopto));
        dffopto = dFF(rngopto,:); dffpreopto = dFF(rngpreopto,:);
        dffs_cp{indtemp} = {dffopto(:,cp), dffpreopto(:,cp)};
        indtemp = indtemp + 1;
    end
    nbins = 90; % bins    
    bin_size = 3;
    opto_tuning = zeros(nbins, 1);
    prevep_tuning = zeros(nbins, 1);
    optodff = dffs_cp{1}{1};
    prevepdff =dffs_cp{1}{2};
    % opto
    time_moving = 1:length(timedFF(rngopto)); % get all times
    ypos_mov = yposopto(time_moving);
    time_moving = time_moving(yposoptomask & trialoptomask);
    ypos_mov = ypos_mov(yposoptomask & trialoptomask);
    for i = 1:nbins
        time_in_bin_opto{i} = time_moving(ypos_mov >= (i-1)*bin_size & ...
            ypos_mov < i*bin_size);
    end
    % pre opto
    time_moving = 1:length(timedFF(rngpreopto)); % get all times
    ypos_mov = ypospreopto(time_moving);
    time_moving = time_moving(ypospreoptomask & trialpreoptomask);
    ypos_mov = ypos_mov(ypospreoptomask & trialpreoptomask);
    for i = 1:nbins
        time_in_bin_pre{i} = time_moving(ypos_mov >= (i-1)*bin_size & ...
            ypos_mov < i*bin_size);
    end

    for bin = 1:nbins
        opto_tuning(bin) = mean(optodff(time_in_bin_opto{bin}));
        prevep_tuning(bin) = mean(prevepdff(time_in_bin_pre{bin}));
    end     

    dffs_cp_dys{mind} = {prevep_tuning,opto_tuning}; % collect tuning curves
    mind=mind+1; dyind=dyind+1;
end
end
%%
meandff_opto_tuning = cell2mat(cellfun(@(x) x{2}, dffs_cp_dys, ...
        'UniformOutput', false));
meandff_prev_tuning = cell2mat(cellfun(@(x) x{1}, dffs_cp_dys, ...
        'UniformOutput', false));
%%
figure;
y = mean(meandff_prev_tuning,2,'omitnan'); % your mean vector;
x = 1:numel(y);
std_dev = std(meandff_prev_tuning','omitnan')';
curve1 = y + std_dev;
curve2 = y - std_dev;
x2 = [x', fliplr(x)'];
inBetween = [curve1, fliplr(curve2)];
fill(x2, inBetween, 'k')%, 'linestyle','none', 'FaceAlpha', 0.2);
hold on;
plot(x, y, 'k', 'LineWidth', 2); hold on;

y = mean(meandff_opto_tuning,2,'omitnan'); % your mean vector;
x = 1:numel(y);
std_dev = std(meandff_opto_tuning','omitnan')';
curve1 = y + std_dev;
curve2 = y - std_dev;
x2 = [x', fliplr(x)'];
inBetween = [curve1, fliplr(curve2)];
fill(x2, inBetween, 'r');%, 'linestyle','none','FaceAlpha', 0.2);
hold on;

plot(x, y, 'r', 'LineWidth', 2); 
