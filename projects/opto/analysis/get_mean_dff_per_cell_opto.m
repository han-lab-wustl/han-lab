% calc opto epoch success and fails
% https://www.nature.com/articles/s41593-022-01050-4
% lick selectivity = smoothed licks in rew zone - smoothed licks in opp
% zone / (smoothed licks in rew zone + smoothed licks in opp zone)
% rewzone = 10 cm before reward
% TODO lick rate outside rew zone
clear all; close all
mouse_name = "e218";
dys = [20,21,22,23,35,36,37,38,39,40,41,42,43,44,45,47 48 49 50 51];
% experiment conditions: preopto=-1; optoep=3/2; control day1=0; control
% day2=1
opto_ep = [-1 -1 -1 -1,3 0 1 2 0 1 3, 0 1 2, 0 3 0 1 2 0]; 
src = "X:\vipcre";
epind = 1; % for indexing
bin_size = 2; % cm
% ntrials = 8; % get licks for last n trials
dffs = {}; % 1 - success opto, 2 - success prev opto, 3 - success postopto, 4 - fails opto, 5 - fails prev opto, 6 - fails prev opto
% other days besides opto : 1 - success, 2 - fails
for dy=dys
    daypth = dir(fullfile(src, mouse_name, string(dy), "**\*Fall.mat"));
    load(fullfile(daypth.folder,daypth.name), 'licks', 'trialnum', 'rewards', 'changeRewLoc', ...
        'ybinned', 'timedFF', 'dFF', 'iscell', 'putative_pcs', 'stat', 'VR');
    pcs = reshape(cell2mat(putative_pcs), [length(putative_pcs{1}), length(putative_pcs)]); % place cell bool
    eps = find(changeRewLoc>0);
    eps = [eps length(changeRewLoc)];
    track_length = 180/VR.scalingFACTOR;
    nbins = track_length/bin_size;
    ybinned = ybinned/VR.scalingFACTOR;
    rewlocs = changeRewLoc(changeRewLoc>0)/VR.scalingFACTOR;
    rewsize = VR.settings.rewardZone/VR.scalingFACTOR;
    if opto_ep(epind)==3
        eprng = eps(3):eps(4);
        trialnum_ = trialnum(eprng);
        reward_ = rewards(eprng);
        licks_ = licks(eprng);
        ybinned_ = ybinned(eprng);
        rewloc = rewlocs(opto_ep(epind));
        [success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum_,reward_);
        success_mask = (ismember(trialnum_,str)); % only do for failed trials
        fails_mask = (ismember(trialnum_,ftr)); % only do for failed trials
        % get 'good' cells
        pc = logical(iscell(:,1));
        [~,bordercells] = remove_border_cells_all_cells(stat, dFF);
        bordercells_pc = bordercells(pc); % mask border cells
        dff = dFF(:,pc); % only iscell
        dff_pc = dff(:,~bordercells_pc); % remove border cells
        dff_pc = dff_pc(eprng,:); 
        dff_pc = dff_pc(:, any(pcs,2)); % onlt place cells
        dff_opto_success = dff_pc(success_mask,:); dff_opto_fails = dff_pc(fails_mask,:);
        % vs. previous epoch
        eprng = eps(2):eps(3);
        trialnum_ = trialnum(eprng);
        reward_ = rewards(eprng);
        licks_ = licks(eprng);
        ybinned_ = ybinned(eprng);
        rewloc = rewlocs(2);
        [success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum_,reward_);
        success_mask = (ismember(trialnum_,str)); % only do for failed trials
        fails_mask = (ismember(trialnum_,ftr)); % only do for failed trials
        % get 'good' cells
        pc = logical(iscell(:,1));
        [~,bordercells] = remove_border_cells_all_cells(stat, dFF);
        bordercells_pc = bordercells(pc); % mask border cells
        dff = dFF(:,pc); % only iscell
        dff_pc = dff(:,~bordercells_pc); % remove border cells
        dff_pc = dff_pc(eprng,:); 
        dff_pc = dff_pc(:, any(pcs,2)); % onlt place cells
        dff_prevopto_success = dff_pc(success_mask,:); dff_prevopto_fails = dff_pc(fails_mask,:);
        
        % vs. next ep
        if length(eps)>4
            try
            eprng = eps(4):eps(5);
            trialnum_ = trialnum(eprng);
            reward_ = rewards(eprng);
            licks_ = licks(eprng);
            ybinned_ = ybinned(eprng);
            rewloc = rewlocs(4);
            [success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum_,reward_);        
            success_mask = (ismember(trialnum_,str)); % only do for failed trials
            fails_mask = (ismember(trialnum_,ftr)); % only do for failed trials
            % get 'good' cells
            pc = logical(iscell(:,1));
            [~,bordercells] = remove_border_cells_all_cells(stat, dFF);
            bordercells_pc = bordercells(pc); % mask border cells
            dff = dFF(:,pc); % only iscell
            dff_pc = dff(:,~bordercells_pc); % remove border cells
            dff_pc = dff_pc(eprng,:); 
            dff_pc = dff_pc(:, any(pcs,2)); % onlt place cells
            dff_postopto_success = dff_pc(success_mask,:); dff_postopto_fails = dff_pc(fails_mask,:);                  
            catch
            end
        else
            dff_postopto_success = 0; dff_postopto_fails=0; 
        end
        dffs{epind} = {dff_prevopto_success, dff_opto_success, dff_postopto_success, ...
            dff_prevopto_fails, dff_opto_fails, dff_postopto_fails};
        
    elseif opto_ep(epind)==2
        eprng = eps(2):eps(3);
        trialnum_ = trialnum(eprng);
        reward_ = rewards(eprng);
        licks_ = licks(eprng);
        ybinned_ = ybinned(eprng);
        rewloc = rewlocs(opto_ep(epind));
        [success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum_,reward_);
        success_mask = (ismember(trialnum_,str)); % only do for failed trials
        fails_mask = (ismember(trialnum_,ftr)); % only do for failed trials
        % get 'good' cells
        pc = logical(iscell(:,1));
        [~,bordercells] = remove_border_cells_all_cells(stat, dFF);
        bordercells_pc = bordercells(pc); % mask border cells
        dff = dFF(:,pc); % only iscell
        dff_pc = dff(:,~bordercells_pc); % remove border cells
        dff_pc = dff_pc(eprng,:); 
        dff_pc = dff_pc(:, any(pcs,2)); % onlt place cells
        dff_opto_success = dff_pc(success_mask,:); dff_opto_fails = dff_pc(fails_mask,:);
        % vs. previous epoch
        eprng = eps(1):eps(2);
        trialnum_ = trialnum(eprng);
        reward_ = rewards(eprng);
        licks_ = licks(eprng);
        ybinned_ = ybinned(eprng);
        rewloc = rewlocs(1);
        [success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum_,reward_);
        success_mask = (ismember(trialnum_,str)); % only do for failed trials
        fails_mask = (ismember(trialnum_,ftr)); % only do for failed trials
        % get 'good' cells
        pc = logical(iscell(:,1));
        [~,bordercells] = remove_border_cells_all_cells(stat, dFF);
        bordercells_pc = bordercells(pc); % mask border cells
        dff = dFF(:,pc); % only iscell
        dff_pc = dff(:,~bordercells_pc); % remove border cells
        dff_pc = dff_pc(eprng,:); 
        dff_pc = dff_pc(:, any(pcs,2)); % onlt place cells
        dff_prevopto_success = dff_pc(success_mask,:); dff_prevopto_fails = dff_pc(fails_mask,:);
        % vs. next ep
        if length(eps)>3
            try
            eprng = eps(3):eps(4);
            trialnum_ = trialnum(eprng);
            reward_ = rewards(eprng);
            licks_ = licks(eprng);
            ybinned_ = ybinned(eprng);
            rewloc = rewlocs(4);
            [success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum_,reward_);
            success_mask = (ismember(trialnum_,str)); % only do for failed trials
            fails_mask = (ismember(trialnum_,ftr)); % only do for failed trials
            % get 'good' cells
            pc = logical(iscell(:,1));
            [~,bordercells] = remove_border_cells_all_cells(stat, dFF);
            bordercells_pc = bordercells(pc); % mask border cells
            dff = dFF(:,pc); % only iscell
            dff_pc = dff(:,~bordercells_pc); % remove border cells
            dff_pc = dff_pc(eprng,:); 
            dff_pc = dff_pc(:, any(pcs,2)); % onlt place cells
            dff_postopto_success = dff_pc(success_mask,:); dff_postopto_fails = dff_pc(fails_mask,:);
            catch % if there are not enough trials etc
            end
        else
            dff_postopto_success = 0; dff_postopto_fails=0;
        end
        dffs{epind} = {dff_prevopto_success, dff_opto_success, dff_postopto_success, ...
            dff_prevopto_fails, dff_opto_fails, dff_postopto_fails};

    elseif opto_ep(epind)==-1 % just pre opto days
        eprng = eps(1):eps(2);
        trialnum_ = trialnum(eprng);
        reward_ = rewards(eprng);
        licks_ = licks(eprng);
        ybinned_ = ybinned(eprng);
        rewloc = rewlocs(1);
        [success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum_,reward_);
        success_mask = (ismember(trialnum_,str)); % only do for failed trials
        fails_mask = (ismember(trialnum_,ftr)); % only do for failed trials
        % get 'good' cells
        pc = logical(iscell(:,1));
        [~,bordercells] = remove_border_cells_all_cells(stat, dFF);
        bordercells_pc = bordercells(pc); % mask border cells
        dff = dFF(:,pc); % only iscell
        dff_pc = dff(:,~bordercells_pc); % remove border cells
        dff_pc = dff_pc(eprng,:); 
        dff_pc = dff_pc(:, any(pcs,2)); % onlt place cells
        dff_preopto_success = dff_pc(success_mask,:); dff_preopto_fails = dff_pc(fails_mask,:);
        dffs{epind} = {dff_preopto_success, dff_preopto_fails};
        
    elseif opto_ep(epind)==0  % intermediate control days 1
        eprng = eps(1):eps(2);
        trialnum_ = trialnum(eprng);
        reward_ = rewards(eprng);
        licks_ = licks(eprng);
        ybinned_ = ybinned(eprng);
        rewloc = rewlocs(1);
        [success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum_,reward_);
        success_mask = (ismember(trialnum_,str)); % only do for failed trials
        fails_mask = (ismember(trialnum_,ftr)); % only do for failed trials
        % get 'good' cells
        pc = logical(iscell(:,1));
        [~,bordercells] = remove_border_cells_all_cells(stat, dFF);
        bordercells_pc = bordercells(pc); % mask border cells
        dff = dFF(:,pc); % only iscell
        dff_pc = dff(:,~bordercells_pc); % remove border cells
        dff_pc = dff_pc(eprng,:); 
        dff_pc = dff_pc(:, any(pcs,2)); % onlt place cells
        dff_inctrl1_success = dff_pc(success_mask,:); dff_inctrl1_fails = dff_pc(fails_mask,:);
        dffs{epind} = {dff_inctrl1_success, dff_inctrl1_fails};
        
    elseif opto_ep(epind)==1  % intermediate control days 2
        eprng = eps(1):eps(2);
        trialnum_ = trialnum(eprng);
        reward_ = rewards(eprng);
        licks_ = licks(eprng);
        ybinned_ = ybinned(eprng);
        rewloc = rewlocs(1);
        [success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum_,reward_);
        success_mask = (ismember(trialnum_,str)); % only do for failed trials
        fails_mask = (ismember(trialnum_,ftr)); % only do for failed trials
        % get 'good' cells
        pc = logical(iscell(:,1));
        [~,bordercells] = remove_border_cells_all_cells(stat, dFF);
        bordercells_pc = bordercells(pc); % mask border cells
        dff = dFF(:,pc); % only iscell
        dff_pc = dff(:,~bordercells_pc); % remove border cells
        dff_pc = dff_pc(eprng,:); 
        dff_pc = dff_pc(:, any(pcs,2)); % onlt place cells
        dff_inctrl2_success = dff_pc(success_mask,:); dff_inctrl2_fails = dff_pc(fails_mask,:);
        dffs{epind} = {dff_inctrl2_success, dff_inctrl2_fails};
    end
    epind = epind+1;
end

% plot certain days of opto vs. ctrl
optoday = dffs{11};
optodayprev_fails = mean(optoday{4},1,'omitnan'); % fails
optodayopto_fails = mean(optoday{5},1,'omitnan');
figure; plot(1,optodayprev_fails,'ko'); hold on; plot(2,optodayopto_fails,'ro'); xlim([0 3])
for ii=1:size(optodayopto_fails,2)
    plot([1, 2],[optodayprev_fails(ii),optodayopto_fails(ii)], 'k'); hold on % pairwise plots
end
[h,p,i,stats] =ttest(optodayprev_fails,optodayopto_fails); % paired