function [dff_opto_success, dff_opto_fails, dff_prevopto_success, ...
    dff_prevopto_fails,dff_postopto_success,dff_postopto_fails] = get_opto_dff(eps, optoep, epind, ...
    trialnum, rewards, licks, ybinned, rewlocs, iscell, stat, dFF, pcs)
    rewsize = 10; % fix hard coded!!!
    eprng = eps(optoep):eps(optoep+1);
    trialnum_ = trialnum(eprng);
    reward_ = rewards(eprng);
    licks_ = licks(eprng);
    ybinned_ = ybinned(eprng);
    rewloc = rewlocs(optoep);
    [success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum_,reward_);
    success_mask = (ismember(trialnum_,str)); % successful trials
    success_mask = success_mask(ybinned_<(rewloc-rewsize/2)); % only get pre reward activity
    fails_mask = (ismember(trialnum_,ftr)); % only do for failed trials
    fails_mask = fails_mask(ybinned_<(rewloc-rewsize/2)); % only get pre reward activity
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
    eprng = eps(optoep-1):eps(optoep);
    trialnum_ = trialnum(eprng);
    reward_ = rewards(eprng);
    licks_ = licks(eprng);
    ybinned_ = ybinned(eprng);
    rewloc = rewlocs(2);
    [success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum_,reward_);
    success_mask = (ismember(trialnum_,str)); % successful trials
    success_mask = success_mask(ybinned_<(rewloc-rewsize/2)); % only get pre reward activity
    fails_mask = (ismember(trialnum_,ftr)); % only do for failed trials
    fails_mask = fails_mask(ybinned_<(rewloc-rewsize/2)); % only get pre reward activity
    % get 'good' cells
    pc = logical(iscell(:,1));
    [~,bordercells] = remove_border_cells_all_cells(stat, dFF);
    bordercells_pc = bordercells(pc); % mask border cells
    dff = dFF(:,pc); % only iscell
    dff_pc = dff(:,~bordercells_pc); % remove border cells
    dff_pc = dff_pc(eprng,:); 
    dff_pc = dff_pc(:, any(pcs,2)); % only place cells
    dff_prevopto_success = dff_pc(success_mask,:); dff_prevopto_fails = dff_pc(fails_mask,:);
    % vs. next ep
    if length(eps)>optoep+1
        try
        eprng = eps(optoep+1):eps(optoep+2);
        trialnum_ = trialnum(eprng);
        reward_ = rewards(eprng);
        licks_ = licks(eprng);
        ybinned_ = ybinned(eprng);
        rewloc = rewlocs(optoep+1);
        [success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum_,reward_);        
        success_mask = (ismember(trialnum_,str)); % successful trials
        success_mask = success_mask(ybinned_<(rewloc-rewsize/2)); % only get pre reward activity
        fails_mask = (ismember(trialnum_,ftr)); % only do for failed trials
        fails_mask = fails_mask(ybinned_<(rewloc-rewsize/2)); % only get pre reward activity
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
end