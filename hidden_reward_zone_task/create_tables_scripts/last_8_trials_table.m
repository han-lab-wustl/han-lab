% tuning curves
close all
clear
clc

settings.paths = dir('/home/gaia/Desktop/E144/E144/D*/Fall.mat');
settings.probe_trials = 'exclude';
settings.bin_size = 5 ; % cm
settings.UL_track = 180;
settings.Fs = 32;
settings.numIterations = 1000;
settings.trials_2compare = 8; %take the last 8 trials of one epoch
settings.I_want2save_figures = true;
settings.I_want2reanlyze = true;

%% ************************************************************************
for this_day = 1:size(settings.paths,1)

    clearvars -except this_day settings
    should_be_analyzed = 1;

    file = fullfile(settings.paths(this_day).folder,settings.paths(this_day).name);
    directory = file;
    info = split(directory,'/');
    mouse_cd = string(info{6});
    day_cd = string(info{7});
    n_trials2compare = settings.trials_2compare;

    folder_table = ['/home/gaia/Desktop/E144/E144/CS_table_last' num2str(n_trials2compare) 'trials'];

    if ~exist(folder_table, 'dir')
        mkdir(folder_table)
    end

    if settings.I_want2reanlyze && this_day == 1
    else
        if ~isempty(dir([folder_table '/cs_table_last' num2str(n_trials2compare) 'trials.mat']))
            CS = load(['/home/gaia/Desktop/E144/E144/CS_table_last' num2str(n_trials2compare) 'trials/cs_table_last' num2str(n_trials2compare) 'trials.mat']);
            this_mouse_is_already_analyzed = cell2mat(cellfun(@(x) strcmp(x,mouse_cd),CS.cs_table.mouse(:),"UniformOutput",false));
            this_day_is_already_analyzed = cell2mat(cellfun(@(x) strcmp(x,day_cd),CS.cs_table.Day(:),"UniformOutput",false));
            should_be_analyzed = exist('CS','var') & (sum(this_mouse_is_already_analyzed + this_day_is_already_analyzed)==2)==0 ;
        end
    end

    if should_be_analyzed
        % load
        load(file)

        disp(['Currently analyzing: ' char(day_cd) ', mouse: ' char(mouse_cd)])
        disp(' ()_()')
        disp("(='.'=)")

        if exist('all','var')

            % extract variables
            probe_trials = settings.probe_trials ;
            bin_size = settings.bin_size ; % cm
            UL_track = settings.UL_track ;
            Fs = settings.Fs ;
            numIterations = settings.numIterations ;
            trials_2compare = settings.trials_2compare ;

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
            end

            reward_locations = changeRewLoc(changeRewLoc(1:epoch_ULs(end))>0);

            nEpoch = numel(reward_locations) ;
            ybinned_relative2reward = nan(1,length_recording);

            n_cells = size(all.dff,1);
            tuning_curves = cell(1,nEpoch);

            nBins = UL_track/bin_size ;

            if nEpoch > 1

                for this_epoch = 1 : nEpoch

                    current_epoch_indeces = epoch_LLs(this_epoch):epoch_ULs(this_epoch);
                    trials_in_this_epoch = trialnum(current_epoch_indeces);
                    ybinned_this_epoch = ybinned(current_epoch_indeces);
                    dff_this_epoch = all.dff(:,current_epoch_indeces);

                    max_trials = max(trials_in_this_epoch);


                    ybinned_last_datapoint = ybinned_this_epoch(end);
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

                        for this_cell = 1 : N_cells
                            x = tuning_curves{EP1 }(this_cell,:) ;
                            y = tuning_curves{EP2 }(this_cell,:) ;

                            cs = getCosineSimilarity(x,y);
                            real_CS{this_comparison}(this_cell,1) = cs;
                            shuffled_CS{this_comparison}{this_cell,1} = nan(1,numIterations);

                            for i = 1 : numIterations
                                random_comparison_cell_index = randperm(n_cells,1);
                                random_y = tuning_curves{EP2} (random_comparison_cell_index,:);
                                cs = getCosineSimilarity(x,random_y);
                                shuffled_CS{this_comparison}{this_cell,1}(i) = cs;
                                shuffled_CS_cell_index{this_comparison}{this_cell,1}(i) = random_comparison_cell_index;
                            end

                            random_cs = shuffled_CS{this_comparison}{this_cell,1};
                            real_cs = real_CS{this_comparison}(this_cell,1);
                            p(this_cell) = sum(random_cs > real_cs)/numIterations ;
                        end

                        comparison_type(this_comparison,:) = [EP1 EP2];
                        tuning_curves_for_this_comparison{this_comparison,1} = tuning_curves{EP1 };
                        tuning_curves_for_this_comparison{this_comparison,2} = tuning_curves{EP2 };


                        % imagesc plot

                        [~,max_bin1] = max(tuning_curves_for_this_comparison{this_comparison,1},[],2);
                        [~,max_bin2] = max(tuning_curves_for_this_comparison{this_comparison,2},[],2);
                        [~,sorted_idx] = sort(max_bin1);

                        TC_imagesc = [tuning_curves_for_this_comparison{this_comparison,1}(sorted_idx,:) tuning_curves_for_this_comparison{this_comparison,2}(sorted_idx,:)];
                        % ----

                        shuffled_distribution = cell2mat(shuffled_CS{this_comparison});
                        shuffled_distribution_count{this_comparison,1} = histcounts...
                            (shuffled_distribution,'Normalization','probability','BinWidth',0.025);

                        real_distribution = real_CS{this_comparison};
                        real_distribution_count{this_comparison,1} = histcounts...
                            (real_distribution,'Normalization','probability','BinWidth',0.025);
                        % [P,H,STATS] = ranksum(real_distribution_count{this_comparison,1},shuffled_distribution_count{this_comparison,1});
                        [P,H,STATS] = ranksum(real_distribution,reshape(shuffled_distribution,[1, numel(shuffled_distribution)]));
                        RankSumP(this_comparison,1) = P;
                        RankSumH(this_comparison,1) = H;
                        RankSumSTATS{this_comparison,1} = STATS;

                        if settings.I_want2save_figures
                            
                            figure('Renderer', 'painters', 'Position', [20 20 1000 700])
                            fig = tiledlayout('flow');
                            nexttile
                            hold on

                            h1 = histogram(shuffled_distribution,'Normalization','probability','BinWidth',0.025);
                            h1.FaceColor = [0 0 0];
                            h1.EdgeColor = [1 1 1];  
                            xline(median(shuffled_distribution,'all'),'-k')

                            h2= histogram(real_distribution,'Normalization','probability','BinWidth',0.025);
                            h2.FaceColor = [1 0 0];
                            h2.EdgeColor = [1 1 1];
                            xline(median(real_distribution,'all'),'-r')

                            axis padded
%                             axis off

                            title(fig,[char(mouse_cd) '; ' char(day_cd)],[ 'ep' num2str(EP1) ' vs. ep' num2str(EP2) '; RankSum p = ' num2str(P) '; last ' num2str(n_trials2compare) ' trials comparison'])
%                             legend({'real',['median real = ' num2str(median(real_distribution,'all'))],...
%                                 'shuffled',['median shuffle = ' num2str(median(shuffled_distribution,'all'))]},'Location','northwest')
                            legend({'shuffled',['median shuffle = ' num2str(median(shuffled_distribution,'all'))],...
                               'real',['median real = ' num2str(median(real_distribution,'all'))]},'Location','northwest')

                            this = nexttile;

                            imagesc(normalize(TC_imagesc ,2,'range'))
                            colormap(turbo)
                            xlabel('cm')
                            xticks([0 max([max_bin1; max_bin2])])
                            xticklabels([0 UL_track])
                            title(this, 'TC last 8 trials',{['epoch ' num2str(EP1) ';epoch ' num2str(EP2)]})

                            axis square

                             this = nexttile;

                             histogram(max_bin1)
                             hold on
                             histogram(max_bin2)
                             title(this, 'Distribution of Max Bin')
                             legend({['epoch ' num2str(EP1)],['epoch ' num2str(EP2)]})
                             ylabel('cell count')
                             xlabel('cm')
                             xticks([0 max([max_bin1; max_bin2])])
                             xticklabels([0 UL_track])
                             axis padded
                             axis square

                             this = nexttile;
                             title(this, 'Max Peak shift')
                             parallelcoords([max_bin1,max_bin2],'Color',[.8 .8 .8])
                             hold on
                             parallelcoords([max_bin1,max_bin2],'quantile',.25,'LineWidth',2)
                             yticks([0 max([max_bin1; max_bin2])])
                             xticklabels({num2str(EP1),num2str(EP2)})
                             yticklabels([0 UL_track])

                             xlabel('Epoch')
                             ylabel('Max Peak')
                             axis padded
                             axis square

                            saveas(fig,['/home/gaia/Desktop/E144/E144/CS_table_last' num2str(n_trials2compare) 'trials/CS_histograms/mat/' char(day_cd) '_ep' num2str(EP1) '_vs_ep' num2str(EP2) 'last' num2str(n_trials2compare) 'trials'],'fig')
                            saveas(fig,['/home/gaia/Desktop/E144/E144/CS_table_last' num2str(n_trials2compare) 'trials/CS_histograms/png/' char(day_cd) '_ep' num2str(EP1) '_vs_ep' num2str(EP2) 'last' num2str(n_trials2compare) 'trials.png'],'png')

                        end
                    end

                    mouse = repmat(mouse_cd, [N_of_epochs2compare, 1]);
                    Day = repmat(day_cd, [N_of_epochs2compare, 1]);
                    shuffled_CS_cell_index = shuffled_CS_cell_index';
                    % HERE
                    cs_table = table...
                        (mouse,Day,comparison_type,tuning_curves_for_this_comparison,real_CS,shuffled_CS,shuffled_CS_cell_index,...
                        real_distribution_count,shuffled_distribution_count,RankSumP,RankSumH,RankSumSTATS);

                    if exist('CS','var') && sum((strcmp(CS.cs_table.Day,day_cd) + strcmp(CS.cs_table.mouse,mouse_cd))==2)==0
                        cs_table = vertcat(CS.cs_table, cs_table);
                        save(['/home/gaia/Desktop/E144/E144/CS_table_last' num2str(n_trials2compare) 'trials/cs_table_last' num2str(n_trials2compare) 'trials'],'cs_table', '-v7.3')
                    else
                        save(['/home/gaia/Desktop/E144/E144/CS_table_last' num2str(n_trials2compare) 'trials/cs_table_last' num2str(n_trials2compare) 'trials'],'cs_table', '-v7.3')
                    end

                end
            end
            disp("('')('')~~~~~~~~~~~*")
            disp('done!')
        end
    end
end
% figure;histogram(p,'BinWidth',0.025)
% xline(0.05,'r','alpha')
%
%
%
% figure;
% histogram(shuffled_distribution,'Normalization','probability','BinWidth',0.025)
% hold on
% histogram(real_distribution,'Normalization','probability','BinWidth',0.025)
%
% figure;
% imagesc(tuning_curves{this_epoch})