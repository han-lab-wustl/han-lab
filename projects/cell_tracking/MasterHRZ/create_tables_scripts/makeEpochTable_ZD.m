function makeEpochTable_ZD(Settings)

addpath("C:\Users\Han\Documents\MATLAB\HANLAB_SCRIPTS2\Analysis") % HARDCODED???

for this_day = 1:size(Settings.paths,1)

    clearvars -except this_day Settings
    should_be_analyzed = 1;

    saving_path = Settings.saving_path;
    file = fullfile(Settings.paths(this_day).folder,Settings.paths(this_day).name);
    directory = file;
    info = split(directory,'\');
    mouse_cd = string(info{Settings.level_mouse_name});
    day_cd = num2str(string(info{Settings.level_day}));
    day=str2num(day_cd(2:end));
    n_trials2compare = Settings.trials_2compare;

    folder_table = [Settings.saving_path num2str(n_trials2compare) 'trials'];

    if ~exist(folder_table, 'dir')
        mkdir(folder_table)
    end

    if should_be_analyzed
        % load
        l = load(file);
        all = l.all; %need to reassign this
        if isfield(l,'remove_iscell')
            remove_iscell = l.remove_iscell;
        end
        ybinned = l.ybinned;

        if exist('all','var')

            % in original script, it checks data to make sure cells that 
            % are not labeled as is cells are removed, do not need here.

            for this_cell = 1 : size(all.dff,1)
                check_this_cell = all.dff(this_cell,all.dff(this_cell,:)<0);
                l.cell2remove(this_cell) = sum(check_this_cell < -10);
            end
            l.cell2remove = l.cell2remove>0;

            if sum(l.cell2remove)>0
                all.dff(l.cell2remove,:) = [];
                all.Fc3(l.cell2remove,:) = [];
                all.Spks(l.cell2remove,:) = [];
            end
            save( file , 'all','-append')
            cell2remove = l.cell2remove;
            save( file , 'cell2remove','-append')

            disp(' ()_()')
            disp("(='.'=)")
            % extract variables
            probe_trials = Settings.probe_trials ;
            bin_size = Settings.bin_size ; % cm
            UL_track = Settings.UL_track ;
            Fs = Settings.Fs ;
            numIterations = Settings.numIterations ;
            trials_2compare = Settings.trials_2compare ;

            length_recording = numel(l.trialnum);
            
            %handles cases if we want to keep probe trials
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

            nEpoch = numel(epoch_LLs) ;
            find(l.changeRewLoc>0,1,"last")
            ybinned_relative2reward = nan(1,length_recording);

            n_cells = size(all.dff,1);
            %init length of tuning curves based on number of epochs in a
            %day
            tuning_curves = cell(1,nEpoch);
                     
            nBins = UL_track/bin_size ;

            if nEpoch >= 1

                for this_epoch = 1 : nEpoch
                       
                    try
                    current_epoch_indeces = epoch_LLs(this_epoch):epoch_ULs(this_epoch);
                    catch
                        keyboard
                    end
                    trials_in_this_epoch = l.trialnum(current_epoch_indeces);
                    ybinned_this_epoch = ybinned(current_epoch_indeces);
                    dff_this_epoch = all.dff(:,current_epoch_indeces);

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

                        tuning_curves{this_epoch} = get_spatial_tuning_all_cells(dff_this_epoch(:,last_Xtrials_indeces)',ybinned_this_epoch(last_Xtrials_indeces),Fs,nBins,UL_track);
                    end

                end

                if numel(tuning_curves) > 1
                    N_cells = size(tuning_curves{1},1)  ;
                    cs_distributions = cell(N_cells, nEpoch );
                    p = nan(N_cells,1);
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

                    for this_comparison = 1 : N_of_epochs2compare

                        EP1 = epochs2compare(this_comparison, 1);
                        EP2 = epochs2compare(this_comparison, 2);
                         %this is what actually calculates the tuning
                         %curves
                        for this_cell = 1 : N_cells
                            x = tuning_curves{EP1 }(this_cell,:) ;
                            y = tuning_curves{EP2 }(this_cell,:) ;

                            if getCosineSimilarity(x,y)<0 % had to move this function/script to 'Analysis'
                                keyboard
                            else

                                cs = getCosineSimilarity(x,y);
                                real_CS{this_comparison}(this_cell,1) = cs;
                                shuffled_CS{this_comparison}{this_cell,1} = nan(1,numIterations);
                            end

                            for i = 1 : numIterations
                                random_comparison_cell_index = randperm(n_cells,1);
                                random_y = tuning_curves{EP2} (random_comparison_cell_index,:);
                                if getCosineSimilarity(x,y)<0
                                    keyboard
                                else
                                    cs = getCosineSimilarity(x,random_y);
                                    shuffled_CS{this_comparison}{this_cell,1}(i) = cs;
                                    shuffled_CS_cell_index{this_comparison}{this_cell,1}(i) = random_comparison_cell_index;
                                end
                            end

                            random_cs = shuffled_CS{this_comparison}{this_cell,1};
                            real_cs = real_CS{this_comparison}(this_cell,1);
                            p(this_cell) = sum(random_cs > real_cs)\numIterations ;
                        end

                        comparison_type(this_comparison,:) = [EP1 EP2];
                        tuning_curves_for_this_comparison{this_comparison,1} = tuning_curves{EP1 };
                        tuning_curves_for_this_comparison{this_comparison,2} = tuning_curves{EP2 };                        
    
                        % imagesc plot
                        %sorts by the maximum firing rate of the each cell
                        [~,max_bin1] = max(tuning_curves_for_this_comparison{this_comparison,1},[],2);
                        [~,max_bin2] = max(tuning_curves_for_this_comparison{this_comparison,2},[],2);
                        [~,sorted_idx] = sort(max_bin1);

                        TC_imagesc = [tuning_curves_for_this_comparison{this_comparison,1}(sorted_idx,:) tuning_curves_for_this_comparison{this_comparison,2}(sorted_idx,:)];
                        % ----
                        if Settings.I_want2save_figures

                            fig = figure('Renderer', 'painters', 'Position', [20 20 1000 700]);                            
                            imagesc(normalize(TC_imagesc,2,'range'))%imagesc(normalize(TC_imagesc,2,'range'))
                            colormap(turbo)
                            xlabel('cm')
                            xticks([0 max([max_bin1; max_bin2])])
                            xticklabels([0 UL_track])
                            yticks(1:size(TC_imagesc,1))
                            yticklabels(sorted_idx) %label with cell id number
                            a = get(gca,'XTickLabel');
                            set(gca,'XTickLabel',a,'FontName','Times','fontsize',8)
                            title(['Day=' day_cd '; TC last 8 trials'],{['epoch ' num2str(EP1) ';epoch ' num2str(EP2)]})                      
                        end
                    end

                    %seperate save settings
                    mouse = repmat(mouse_cd, [N_of_epochs2compare, 1]);
                    Day = repmat(day, [N_of_epochs2compare, 1]);
                    shuffled_CS_cell_index = shuffled_CS_cell_index';
                    % HERE
                    cs_table = table...
                        (mouse,Day,comparison_type,tuning_curves_for_this_comparison,real_CS,shuffled_CS,shuffled_CS_cell_index,...
                        real_distribution_count,shuffled_distribution_count,RankSumP,RankSumH,RankSumSTATS);

                    if exist('CS','var') && sum((strcmp(CS.cs_table.Day,day_cd) + strcmp(CS.cs_table.mouse,mouse_cd))==2)==0
                        cs_table = vertcat(CS.cs_table, cs_table);
                        save([Settings.saving_path num2str(n_trials2compare) 'trials\cs_table_last' num2str(n_trials2compare) 'trials'],'cs_table', '-v7.3')
                    else
                        save([Settings.saving_path num2str(n_trials2compare) 'trials\', ...
                            sprintf('%s_d%03d_cs_table_last', mouse_cd, day) num2str(n_trials2compare) 'trials'], 'cs_table', '-v7.3')
                    end

                else % for days with only 1 epoch
                    N_cells = size(tuning_curves{1},1)  ;
                    EP1 = 1;
                    %this is what actually calculates the tuning
                    %curves?
                    %removed shuffle comparisons etc
   
                    % imagesc plot
                    %sorts by the maximum firing rate of the each cell
                    [~,max_bin1] = max(tuning_curves{1},[],2);
                    [~,sorted_idx] = sort(max_bin1);

                    TC_imagesc = tuning_curves{1}(sorted_idx,:);
                        % ----
                    if Settings.I_want2save_figures
                        fig = figure('Renderer', 'painters', 'Position', [20 20 1000 700]);                            
                        imagesc(normalize(TC_imagesc,2,'range'))%imagesc(normalize(TC_imagesc,2,'range'))
                        colormap(turbo)
                        xlabel('cm')
                        xticks([0 max(max_bin1)])
                        xticklabels([0 UL_track])
                        yticks(1:size(TC_imagesc,1))
                        yticklabels(sorted_idx) %label with cell id number
                        a = get(gca,'XTickLabel');
                        set(gca,'XTickLabel',a,'FontName','Times','fontsize',8)
                        title(['Day=' day_cd '; TC last 8 trials'],{['epoch ' num2str(EP1)]})                      
                    end
                    %save just 1 tuning curve
                    %seperate save settings
                    mouse = repmat(mouse_cd, [1, 1]);
                    Day = repmat(day, [1, 1]);
                    comparison_type=[1 1];
                    % HERE; just for 1 epoch
                    cs_table = table...
                        (mouse,Day,comparison_type,tuning_curves);

                    if exist('CS','var') && sum((strcmp(CS.cs_table.Day,day_cd) + strcmp(CS.cs_table.mouse,mouse_cd))==2)==0
                        cs_table = vertcat(CS.cs_table, cs_table);
                        save([Settings.saving_path num2str(n_trials2compare) 'trials\cs_table_last' num2str(n_trials2compare) 'trials'],'cs_table', '-v7.3')
                    else
                        save([Settings.saving_path num2str(n_trials2compare) 'trials\', ...
                            sprintf('%s_d%03d_cs_table_last', mouse_cd, day) num2str(n_trials2compare) 'trials'], 'cs_table', '-v7.3')
                    end
                end                    

                end
            end
            disp('done!')
      end
end

end

