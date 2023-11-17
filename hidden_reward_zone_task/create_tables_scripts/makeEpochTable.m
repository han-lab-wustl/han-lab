function makeEpochTable(Settings)

for this_day = 1:size(Settings.paths,1)% this should call a singular function per day

    clearvars -except this_day Settings

    file = fullfile(Settings.paths(this_day).folder,Settings.paths(this_day).name);
    directory = file;
    mouse_cd = Settings.paths(this_day).name; % in zahra's folder, mouse name and day in fall name
    n_trials2compare = Settings.trials_2compare;

    folder_table = [Settings.saving_path num2str(n_trials2compare) 'trials'];

    if ~exist(folder_table, 'dir')
        mkdir(folder_table)
    end

    if Settings.I_want2reanlyze && this_day == 1
    else
        if ~isempty(dir([folder_table '\cs_table_last' num2str(n_trials2compare) 'trials.mat']))
            CS = load([Settings.saving_path num2str(n_trials2compare) 'trials\cs_table_last' num2str(n_trials2compare) 'trials.mat']);
            %this_mouse_is_already_analyzed = cell2mat(cellfun(@(x) strcmp(x,mouse_cd),CS.cs_table.mouse(:),"UniformOutput",false));
            %this_day_is_already_analyzed = cell2mat(cellfun(@(x) strcmp(x,day_cd),CS.cs_table.Day(:),"UniformOutput",false));
            %should_be_analyzed = exist('CS','var') & (sum(this_mouse_is_already_analyzed + this_day_is_already_analyzed)==2)==0 ;
            this_day_already_exist = cellfun(@(x) strcmp(x,day_cd),string(CS.cs_table.Day),'UniformOutput',false);
            this_mouse_already_exist = cellfun(@(x) strcmp(x,mouse_cd),string(CS.cs_table.mouse),'UniformOutput',false);
            should_be_analyzed = exist('CS','var') & ...
                sum((cell2mat(this_day_already_exist)...
                + cell2mat(this_mouse_already_exist))==2)==0 ;
        end
    end

    % load
    l = load(file);
    ybinned = l.ybinned/Settings.gainVR; % divide by gain MISSING IN MASTERHRZ, zd added
    % make fake 'all' vars with zahra's iscell condition only
    all.dff = l.dFF(:,logical(l.iscell(:,1)));
    all.fc3 = l.Fc3(:,logical(l.iscell(:,1)));
    forwardvel=l.forwardvel; % we need to use this to calculcate moving time

    disp(['Currently analyzing: mouse: ' char(mouse_cd)])
    disp(' ()_()')
    disp("(='.'=)")

    % extract variables
    probe_trials = Settings.probe_trials ;
    bin_size = Settings.bin_size ; % cm
    UL_track = Settings.UL_track ;
    Fs = Settings.Fs ; % normalize by planes you idiot
    numIterations = Settings.numIterations ;
    trials_2compare = Settings.trials_2compare ;

    length_recording = numel(l.trialnum);

    switch probe_trials

        case 'exclude'
            first_rewarded_trial = find(l.trialnum >= 3,1) ;
            single_trials_boundaries = [first_rewarded_trial diff(l.trialnum) == 1];
            first_epoch_trials = l.trialnum == 3;
            if l.trialnum(1)>3
                first_epoch_trials(1) = l.trialnum(1)>3;
            end
            epoch_LLs = find(single_trials_boundaries & first_epoch_trials);
            last_trial_is_a_probe_trial = l.trialnum(length_recording) < 3;
            if last_trial_is_a_probe_trial
                epoch_ULs = find(diff(l.trialnum)< -3);
            else
                epoch_ULs = [find(diff(l.trialnum)< -3) length_recording];
            end

        case 'end'
            first_rewarded_trial = find(l.trialnum >= 3,1) ;
            single_trials_upBoundaries = [diff(l.trialnum) == 1 1] ;
            single_trials_lowBoundaries = [first_rewarded_trial diff(l.trialnum) == 1] ;
            last_probe_trials = l.trialnum == 2 ;
            first_epoch_trials = l.trialnum == 3 ;
            epoch_LLs = find(single_trials_lowBoundaries & first_epoch_trials) ;
            epoch_ULs = find(last_probe_trials & single_trials_upBoundaries) ;

        case 'beginning'
            epoch_LLs = [1 find(diff(l.trialnum) <0)+1] ;
            epoch_ULs =[find(diff(l.trialnum)< 0) numel(l.trialnum)] ;
    end

    reward_locations = l.changeRewLoc(l.changeRewLoc(1:epoch_ULs(end))>0);
    nEpoch = numel(epoch_LLs);
    find(l.changeRewLoc>0,1,"last");
    tuning_curves = cell(1,nEpoch);
    nBins = UL_track/bin_size;

    if nEpoch > 1
        for this_epoch = 1:nEpoch
            try
                current_epoch_indeces = epoch_LLs(this_epoch):epoch_ULs(this_epoch);
            catch
                keyboard
            end
            % doesn't get successful trials only, greattt
            trials_in_this_epoch = l.trialnum(current_epoch_indeces);
            ybinned_this_epoch = ybinned(current_epoch_indeces);
            dff_this_epoch = all.fc3(current_epoch_indeces, :); % everything done in fc3 not dff like ele's version
            forwardvel_ = forwardvel(current_epoch_indeces);
            max_trials = max(trials_in_this_epoch);
            try
                ybinned_last_datapoint = ybinned_this_epoch(end);
            catch
                keyboard
            end
            ndatapoints_last_trial = sum(trials_in_this_epoch==max_trials);
            we_shall_keep_the_last_trial = ybinned_last_datapoint > (UL_track-10) && ndatapoints_last_trial > 10;
            if ~we_shall_keep_the_last_trial
                max_trials = max_trials -1;
            end
            min_trials = max_trials-trials_2compare;
            if min_trials > 0
                last_Xtrials_indeces = trials_in_this_epoch>min_trials & trials_in_this_epoch<=max_trials;
                tuning_curves{this_epoch} = get_spatial_tuning_all_cells(dff_this_epoch(last_Xtrials_indeces,:)', ...
                    forwardvel_(last_Xtrials_indeces), ...
                    ybinned_this_epoch(last_Xtrials_indeces),Fs,nBins,UL_track);
            else
                fprintf('\n *******not enough successful trials in ep%i *******', this_epoch)
            end
        end

        N_cells = size(tuning_curves{1},1)  ;
        % init vars
        epochs2compare = nchoosek(1:sum(cellfun(@(x) ~isempty(x),tuning_curves)),2);
        N_of_epochs2compare = size(epochs2compare ,1);
        comparison_type = nan(N_of_epochs2compare,2);
        shuffled_CS = cell(N_of_epochs2compare,1);
        real_CS = cell(N_of_epochs2compare,1);
        shuffled_distribution_count = cell(N_of_epochs2compare,1);
        real_distribution_count = cell(N_of_epochs2compare,1);
        RankSumP = nan(N_of_epochs2compare,1);
        RankSumH = nan(N_of_epochs2compare,1);
        RankSumSTATS = cell(N_of_epochs2compare,1);
        tuning_curves_for_this_comparison = cell(N_of_epochs2compare,1);
        shuffled_CS_cell_index = cell(N_of_epochs2compare,1);
        % PLEASE MAKE INTO FUNCTION
        % just updated init variables per comparison
        for this_comparison = 1:N_of_epochs2compare
            [comparison_type, tuning_curves_for_this_comparison,...
                real_CS,shuffled_CS,shuffled_CS_cell_index,...
            real_distribution_count,shuffled_distribution_count,RankSumP,RankSumH,RankSumSTATS] = plots_per_ep_comparison(epochs2compare, ...
                this_comparison, tuning_curves, tuning_curves_for_this_comparison, numIterations, N_cells, ...
                comparison_type, shuffled_CS, real_CS, shuffled_CS_cell_index, ...
                shuffled_distribution_count, real_distribution_count, ...
                RankSumP, RankSumH, RankSumSTATS, mouse_cd, n_trials2compare, ...
                UL_track, reward_locations, Settings);
        end

        mouse = repmat(mouse_cd, [N_of_epochs2compare, 1]);        
        shuffled_CS_cell_index = shuffled_CS_cell_index';
        % HERE
        cs_table = table...
            (mouse,comparison_type,tuning_curves_for_this_comparison,real_CS,shuffled_CS,shuffled_CS_cell_index,...
            real_distribution_count,shuffled_distribution_count,RankSumP,RankSumH,RankSumSTATS);

        if exist('CS','var') && sum((strcmp(CS.cs_table.Day,day_cd) + strcmp(CS.cs_table.mouse,mouse_cd))==2)==0
            cs_table = vertcat(CS.cs_table, cs_table);
            save([Settings.saving_path num2str(n_trials2compare) 'trials_cs_table'],'cs_table', '-v7.3')
        else
            save([Settings.saving_path num2str(n_trials2compare) 'trials_cs_table'],'cs_table', '-v7.3')
        end
    end
    disp("('')('')~~~~~~~~~~~*")
    disp('done!')

end
end
