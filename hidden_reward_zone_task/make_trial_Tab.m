%% Trial by trial changes in neuronal activity
function make_trial_Tab(Settings)
%%
for this_day = 1:size(Settings.paths,1)

    clearvars -except this_day Settings
    should_be_analyzed = 1;

    file = fullfile(Settings.paths(this_day).folder,Settings.paths(this_day).name);
    directory = file;
    info = split(directory,'\');
    mouse_cd = string(info{Settings.level_mouse_name});
    day_cd = string(info{Settings.level_day});


    folder_table = [Settings.saving_path 'CS_table_all_trials'];

    if ~exist(folder_table, 'dir')
        mkdir(folder_table)
    end

    if Settings.I_want2reanlyze && this_day == 1
    else
        if ~isempty(dir(folder_table ))
            CS_trial_by_trial_probes_all = load([folder_table '\CS_trial_by_trial_probes_all.mat']);
            this_day_already_exist = cellfun(@(x) strcmp(x,day_cd),string(CS_trial_by_trial_probes_all.table_trial_by_trial_probes_all_CS.trial_day),'UniformOutput',false);
            this_mouse_already_exist = cellfun(@(x) strcmp(x,mouse_cd),string(CS_trial_by_trial_probes_all.table_trial_by_trial_probes_all_CS.mouse),'UniformOutput',false);
            should_be_analyzed = exist('CS_trial_by_trial_probes_all','var') & ...
                sum((cell2mat(this_day_already_exist)...
                + cell2mat(this_mouse_already_exist))==2)==0 ;
        end
    end

    if should_be_analyzed

        % load
        load(file);



        disp(['Currently analyzing: ' char(day_cd) ', mouse: ' char(mouse_cd)])
        disp(' ()_()')
        disp("(='.'=)")

        if exist('all','var')

            % extract variables
            probe_trials = Settings.probe_trials ;
            bin_size = Settings.bin_size ; % cm
            UL_track = Settings.UL_track ;
            Fs = Settings.Fs ;
            numIterations = Settings.numIterations ;

            length_recording = numel(trialnum);


            switch probe_trials

                case 'exclude'
                    first_rewarded_trial = find(trialnum >= 3,1) ;
                    single_trials_boundaries = [first_rewarded_trial diff(trialnum) == 1];
                    first_epoch_trials = trialnum == 3;
                    epoch_LLs = find(single_trials_boundaries & first_epoch_trials);
                    last_trial_is_a_probe_trial = trialnum(length_recording) < 3;
                    if last_trial_is_a_probe_trial
                        epoch_ULs = find(diff(trialnum)< 0);
                    else
                        epoch_ULs = [find(diff(trialnum)< 0) length_recording];
                    end

                case 'end'
                    first_rewarded_trial = find(trialnum >= 3,1) ;
                    single_trials_upBoundaries = [diff(trialnum) == 1 1] ;
                    single_trials_lowBoundaries = [first_rewarded_trial diff(trialnum) == 1] ;
                    last_probe_trials = trialnum == 2 ;
                    first_epoch_trials = trialnum == 3 ;
                    epoch_LLs = find(single_trials_lowBoundaries & first_epoch_trials) ;
                    epoch_ULs = find(last_probe_trials & single_trials_upBoundaries) ;

                case 'beginning'
                    epoch_LLs = [1 find(diff(trialnum) <0)+1] ;
                    epoch_ULs =[find(diff(trialnum)< 0) numel(trialnum)] ;

                case 'intra_all'
                    epoch_starts = diff(trialnum)<0;
                    unique_trialnum_idx = diff(trialnum) ;
                    unique_trialnum = trialnum + 1 ;
                    unique_trialnum (unique_trialnum_idx < 0 | unique_trialnum_idx > 0) = 0;
                    unique_trialnum = bwlabel(unique_trialnum);
                    %                     probes_in_unique_trialnum = unique_trialnum(trialnum < 3);
                    %                     identity_probes_in_unique_trialnum = unique(probes_in_unique_trialnum);
                    %                     index_identity_probes_edges_UL = [diff(identity_probes_in_unique_trialnum)>1 0];
                    %                     index_identity_probes_edges_LL = diff([0 identity_probes_in_unique_trialnum])>1;
                    %                     identity_probes_edges = identity_probes_in_unique_trialnum((index_identity_probes_edges_UL + index_identity_probes_edges_LL)==1);
                    %                     unique_probes = identity_probes_edges (diff(identity_probes_edges)==2);

                    %                     trials_we_want2_study = {};
                    %                     for i = 1 : numel(unique_probes)
                    %                         LL = unique_probes(i) - Settings.probe_Lower;
                    %                         UL = 2 + unique_probes(i) + Settings.probe_Upper;
                    %                         trials_we_want2_study{i} = LL : UL;
                    %                     end
                    %                     trials_we_want2_study = cell2mat(trials_we_want2_study);

                    trials_we_want2_study = unique(unique_trialnum);
                    trials_we_want2_study(trials_we_want2_study==0)=[];
                    %                     trials_we_want2_study = intersect(trials_we_want2_study,identity_trials_unique_trialnum);

            end
            if ~isempty(trials_we_want2_study)
                epochs = trialnum +1 ;
                epochs(diff(trialnum)< 0 ) = 0;
                epochs = bwlabel(epochs);

                attachEpoch = zeros(1,numel(trialnum));
                for this_epoch = 1 : max(epochs)
                    index_this_epoch = epochs == this_epoch;
                    max_trialnum_this_epoch = max(trialnum(index_this_epoch));
                    there_are_probe_trials_only = max_trialnum_this_epoch < 3;
                    if there_are_probe_trials_only
                        last_index_this_epoch = max(find(index_this_epoch)+1);
                        attachEpoch(last_index_this_epoch) = 1;
                    end
                end

                attachEpoch = attachEpoch(1:numel(trialnum));
                epochs = trialnum +1 ; % correct for 'double probe trials' during remap
                indexEpoch = diff(trialnum)< 0 ;
                indexEpoch(attachEpoch==1) = 0;
                epochs(indexEpoch) = 0;
                epochs = epochs+attachEpoch;
                epochs = bwlabel(epochs);

                if sum(attachEpoch)>0
                    figure;
                    plot(normalize(trialnum,'range'))
                    hold on
                    plot(normalize(epochs,'range'))
                    legend({'trialnum','epochs'})

                    drawnow
                end

                nBins = UL_track/bin_size ;

                remove_idx = 0;
                for current_trial = trials_we_want2_study
                    remove_idx = remove_idx+1;
                    remove_trial_index(remove_idx) = false;
                    this_trial = unique_trialnum == current_trial; % trail index
                    if sum(this_trial) > 10

                        yposition = ybinned(this_trial);
                        track_completed = sum(yposition>160)>4;


                        if track_completed % there are some indeces ...
                            ypos{current_trial} = ybinned(this_trial);
                            real_trial(current_trial,1) = mean(trialnum(this_trial));
                            dff{current_trial} = all.dff(:,this_trial);
                            cs_trial_by_trial_epoch(current_trial) = mean(epochs(this_trial));
                            tuning_curves{current_trial} = get_spatial_tuning_all_cells...
                                (dff{current_trial}',ypos{current_trial},Fs,nBins,UL_track);


                            cs_trial_by_trial_mouse{current_trial,1} = mouse_cd ;
                            cs_trial_by_trial_day{current_trial,1}= day_cd ;
                            reward_location(current_trial) = sum(changeRewLoc(this_trial));
                        else
                            remove_trial_index(remove_idx) = true;
                        end
                    else
                        remove_trial_index(remove_idx) = true;
                    end

                end
                trials_we_want2_study(remove_trial_index) = [];

                for this_trial = 1 : numel(trials_we_want2_study)

                    if this_trial ~= numel(trials_we_want2_study)
                        T1 = trials_we_want2_study(this_trial);
                        real_T1 = real_trial(T1);
                        T2 = trials_we_want2_study(this_trial+1);
                        real_T2 = real_trial(T2);
                        x_mat = tuning_curves{T1};
                        x = reshape(x_mat,[1,numel(x_mat)]);
                        x(isnan(x)) = [];
                        y_mat = tuning_curves{T2};
                        y = reshape(y_mat,[1,numel(y_mat)]);
                        y(isnan(y)) = [];

                        try
                            if getCosineSimilarity(x,y)<0
                                keyboard
                            else
                                cs_trial_by_trial{T1,1} = getCosineSimilarity(x,y);
                                cs_trial_by_trial_comparison(T1,1:2) = [real_T1 real_T2];
                            end
                        catch
                            figure;
                            imagesc([x_mat y_mat])
                            title('one incomplete trial in this comparison')
                            drawnow
                            cs_trial_by_trial{T1,1} = [];
                            cs_trial_by_trial_comparison(T1,1:2) = [real_T1 real_T2];

                        end
                    else
                        T1 = trials_we_want2_study(this_trial);
                        real_T1 = real_trial(T1);
                        T2 = nan;
                        real_T2 = nan;
                        cs_trial_by_trial{T1,1} = [];
                        cs_trial_by_trial_comparison(T1,1:2) = [real_T1 real_T2];
                    end


                end


                trial_day = cs_trial_by_trial_day;
                mouse = cs_trial_by_trial_mouse;
                tuning_curves = tuning_curves';
                pop_CS = cs_trial_by_trial;
                trial_comparison = cs_trial_by_trial_comparison;
                trial_epoch = cs_trial_by_trial_epoch';

                reward_location = reward_location';
                rL = changeRewLoc(changeRewLoc>0);
                unique_epochs = unique(trial_epoch)';
                if sum(unique_epochs == 0 ) >0
                    unique_epochs(unique_epochs == 0 ) = [];
                end
                id = 1;
                for i = unique_epochs
                    indeces = find(trial_epoch==i);
                    if numel(indeces)>3
                        reward_location(indeces) = rL(id);
                        id = id+1;
                    else
                        reward_location(indeces) = 0;
                    end
                end

                rew_location = reward_location;
                dff = dff';
                ypos = ypos';


                table_trial_by_trial_probes_all_CS = table (mouse,trial_day,tuning_curves,...
                    pop_CS,trial_comparison,trial_epoch,rew_location,dff,ypos);
                table_trial_by_trial_probes_all_CS(table_trial_by_trial_probes_all_CS.trial_epoch==0,:) = [];

                %                 if Settings.I_want2save_figures
                %                     fig = figure;
                %                     for this_epoch = 1 : nEpoch
                %                         y = cell2mat(cs_trial_by_trial{this_epoch});
                %                         x = 1: numel(cs_trial_by_trial{this_epoch});
                %
                %                         for this_cell = 1 : size(y,2)
                %                             cs_this_cell_across_trial_comparisons = y(:,this_cell);
                %                             x = find(cs_this_cell_across_trial_comparisons);
                %                             scatter(x,1-cs_this_cell_across_trial_comparisons,'filled','c')
                %                             hold on
                %                         end
                %                     end
                %
                %                     title([char(mouse_cd) '; ' char(day_cd)], 'population difference CS')
                %
                %                     saveas(fig,['\home\gaia\Desktop\CS_trial_by_trial_probes_all\mat\' char(day_cd) ],'fig')
                %                     saveas(fig,['\home\gaia\Desktop\CS_trial_by_trial_probes_all\png\' char(day_cd) '.png'],'png')
                %
                %                 end


                if exist('CS_trial_by_trial_probes_all','var')
                    table_trial_by_trial_probes_all_CS = vertcat(CS_trial_by_trial_probes_all.table_trial_by_trial_probes_all_CS, table_trial_by_trial_probes_all_CS );
                    save([folder_table '\CS_trial_by_trial_probes_all'],'table_trial_by_trial_probes_all_CS','-v7.3')
                else
                    save([folder_table '\CS_trial_by_trial_probes_all'],'table_trial_by_trial_probes_all_CS','-v7.3')
                end

            end
        end
    end

end

