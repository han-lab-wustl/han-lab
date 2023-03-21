% tuning curves
close all
clear
clc

Settings.paths = dir('/home/gaia/Desktop/E144/E144/D*/Fall.mat');
Settings.probe_trials = 'exclude';
Settings.bin_size = 5 ; % cm
Settings.UL_track = 180;
Settings.Fs = 32;
Settings.numIterations = 1000;
Settings.trials_2compare = 8; %take the last 8 trials of one epoch
Settings.I_want2save_figures = true;
Settings.I_want2reanlyze = true;

%% ************************************************************************
for this_day = 1:size(Settings.paths,1)

    clearvars -except this_day Settings
    should_be_analyzed = 1;

    file = fullfile(Settings.paths(this_day).folder,Settings.paths(this_day).name);
    directory = file;
    info = split(directory,'/');
    mouse_cd = string(info{6});
    day_cd = string(info{7});
    n_trials2compare = Settings.trials_2compare;

    folder_table = ['/home/gaia/Desktop/E144/E144/ND_table_last' num2str(n_trials2compare) 'trials'];

    if ~exist(folder_table, 'dir')
        mkdir(folder_table)
    end

    if Settings.I_want2reanlyze && this_day == 1
    else
        if ~isempty(dir([folder_table '/nd_table_last' num2str(n_trials2compare) 'trials.mat']))
            ND = load(['/home/gaia/Desktop/E144/E144/ND_table_last' num2str(n_trials2compare) 'trials/nd_table_last' num2str(n_trials2compare) 'trials.mat']);
            %this_mouse_is_already_analyzed = cell2mat(cellfun(@(x) strcmp(x,mouse_cd),ND.nd_table.mouse(:),"UniformOutput",false));
            %this_day_is_already_analyzed = cell2mat(cellfun(@(x) strcmp(x,day_cd),ND.nd_table.Day(:),"UniformOutput",false));
            %should_be_analyzed = exist('ND','var') & (sum(this_mouse_is_already_analyzed + this_day_is_already_analyzed)==2)==0 ;
            this_day_already_exist = cellfun(@(x) strcmp(x,day_cd),string(ND.nd_table.Day),'UniformOutput',false);
            this_mouse_already_exist = cellfun(@(x) strcmp(x,mouse_cd),string(ND.nd_table.mouse),'UniformOutput',false);
            should_be_analyzed = exist('ND','var') & ...
                sum((cell2mat(this_day_already_exist)...
                + cell2mat(this_mouse_already_exist))==2)==0 ;
        end
    end

    if should_be_analyzed
        % load
        load(file)
        if exist('all','var')
        
        % check data
        if exist('remove_iscell','var')
        all.dff(remove_iscell==1,:) = [];
        all.Fc3(remove_iscell==1,:) = [];
        all.Spks(remove_iscell==1,:) = [];
        end

        for this_cell = 1 : size(all.dff,1)
        check_this_cell = all.dff(this_cell,all.dff(this_cell,:)<0);
        cell2remove(this_cell) = sum(check_this_cell < -10);
        end
        cell2remove = cell2remove>0;

        if sum(cell2remove)>0
            all.dff(cell2remove,:) = [];
            all.Fc3(cell2remove,:) = [];
            all.Spks(cell2remove,:) = [];
        end
        save( file , 'all','-append') 
        save( file , 'cell2remove','-append') 


        disp(['Currently analyzing: ' char(day_cd) ', mouse: ' char(mouse_cd)])
        disp(' ()_()')
        disp("(='.'=)")


            % extract variables
            probe_trials = Settings.probe_trials ;
            bin_size = Settings.bin_size ; % cm
            UL_track = Settings.UL_track ;
            Fs = Settings.Fs ;
            numIterations = Settings.numIterations ;
            trials_2compare = Settings.trials_2compare ;

            length_recording = numel(trialnum);


            switch probe_trials

                case 'exclude'
                    first_rewarded_trial = find(trialnum >= 3,1) ;
                    single_trials_boundaries = [first_rewarded_trial diff(trialnum) == 1];
                    first_epoch_trials = trialnum == 3;
                    epoch_LLs = find(single_trials_boundaries & first_epoch_trials);
                    last_trial_is_a_probe_trial = trialnum(length_recording) < 3;
                    if last_trial_is_a_probe_trial
                        epoch_ULs = find(diff(trialnum)< -3);
                    else
                        epoch_ULs = [find(diff(trialnum)< -3) length_recording];
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
            find(changeRewLoc>0,1,"last") 
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
                    nd_distributions = cell(N_cells, nEpoch );
                    p = nan(N_cells,1);
                    epochs2compare = nchoosek(1:sum(cellfun(@(x) ~isempty(x),tuning_curves)),2);
                    N_of_epochs2compare = size(epochs2compare ,1);
                    comparison_type = nan(N_of_epochs2compare,2);
                    shuffled_ND = cell(N_of_epochs2compare,1);
                    real_ND = cell(N_of_epochs2compare,1);
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

                           

                                nd = sum(normalize(x)-normalize(y));
                                real_ND{this_comparison}(this_cell,1) = nd;
                                shuffled_ND{this_comparison}{this_cell,1} = nan(1,numIterations);
                          

                            for i = 1 : numIterations
                                random_comparison_cell_index = randperm(n_cells,1);
                                random_y = tuning_curves{EP2} (random_comparison_cell_index,:);

                                    nd = sum((normalize(x)-normalize(random_y)));
                                    shuffled_ND{this_comparison}{this_cell,1}(i) = nd;
                                    shuffled_ND_cell_index{this_comparison}{this_cell,1}(i) = random_comparison_cell_index;
                              
                            end

                            random_nd = shuffled_ND{this_comparison}{this_cell,1};
                            real_nd = real_ND{this_comparison}(this_cell,1);
                            p(this_cell) = sum(random_nd > real_nd)/numIterations ;
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

                        shuffled_distribution = cell2mat(shuffled_ND{this_comparison});
                        shuffled_distribution_count{this_comparison,1} = histcounts...
                            (shuffled_distribution,'Normalization','probability','BinWidth',0.025);

                        real_distribution = real_ND{this_comparison};
                        real_distribution_count{this_comparison,1} = histcounts...
                            (real_distribution,'Normalization','probability','BinWidth',0.025);
                        % [P,H,STATS] = ranksum(real_distribution_count{this_comparison,1},shuffled_distribution_count{this_comparison,1});
                        [P,H,STATS] = ranksum(real_distribution,reshape(shuffled_distribution,[1, numel(shuffled_distribution)]));
                        RankSumP(this_comparison,1) = P;
                        RankSumH(this_comparison,1) = H;
                        RankSumSTATS{this_comparison,1} = STATS;

                        if Settings.I_want2save_figures

                            figure('Renderer', 'painters', 'Position', [20 20 1000 700])
                            fig = tiledlayout('flow');
                            nexttile
                            hold on
        
                            h1 = histogram(shuffled_distribution,'Normalization','probability','BinWidth',0.25);
                            h1.FaceColor = [0 0 0];
                            h1.EdgeColor = 'none';
                            xline(median(shuffled_distribution,'all'),'-k')

                            h2= histogram(real_distribution,'Normalization','probability','BinWidth',0.25);
                            h2.FaceColor = [1 0 0];
                            h2.EdgeColor = 'none';
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

                            try

                            saveas(fig,['/home/gaia/Desktop/E144/E144/ND_table_last' num2str(n_trials2compare) 'trials/ND_histograms/mat/' char(day_cd) '_ep' num2str(EP1) '_vs_ep' num2str(EP2) 'last' num2str(n_trials2compare) 'trials'],'fig')
                            saveas(fig,['/home/gaia/Desktop/E144/E144/ND_table_last' num2str(n_trials2compare) 'trials/ND_histograms/png/' char(day_cd) '_ep' num2str(EP1) '_vs_ep' num2str(EP2) 'last' num2str(n_trials2compare) 'trials.png'],'png')
                            saveas(fig,['/home/gaia/Desktop/E144/E144/ND_table_last' num2str(n_trials2compare) 'trials/ND_histograms/svg/' char(day_cd) '_ep' num2str(EP1) '_vs_ep' num2str(EP2) 'last' num2str(n_trials2compare) 'trials.svg'],'svg')

                            catch

                            mkdir(['/home/gaia/Desktop/E144/E144/ND_table_last' num2str(n_trials2compare) 'trials/ND_histograms/mat/'])
                            mkdir(['/home/gaia/Desktop/E144/E144/ND_table_last' num2str(n_trials2compare) 'trials/ND_histograms/png/'])
                            mkdir(['/home/gaia/Desktop/E144/E144/ND_table_last' num2str(n_trials2compare) 'trials/ND_histograms/svg/'])

                            saveas(fig,['/home/gaia/Desktop/E144/E144/ND_table_last' num2str(n_trials2compare) 'trials/ND_histograms/mat/' char(day_cd) '_ep' num2str(EP1) '_vs_ep' num2str(EP2) 'last' num2str(n_trials2compare) 'trials'],'fig')
                            saveas(fig,['/home/gaia/Desktop/E144/E144/ND_table_last' num2str(n_trials2compare) 'trials/ND_histograms/png/' char(day_cd) '_ep' num2str(EP1) '_vs_ep' num2str(EP2) 'last' num2str(n_trials2compare) 'trials.png'],'png')
                            saveas(fig,['/home/gaia/Desktop/E144/E144/ND_table_last' num2str(n_trials2compare) 'trials/ND_histograms/svg/' char(day_cd) '_ep' num2str(EP1) '_vs_ep' num2str(EP2) 'last' num2str(n_trials2compare) 'trials.svg'],'svg')

                            end
                        end
                    end

                    mouse = repmat(mouse_cd, [N_of_epochs2compare, 1]);
                    Day = repmat(day_cd, [N_of_epochs2compare, 1]);
                    shuffled_ND_cell_index = shuffled_ND_cell_index';
                    % HERE
                    nd_table = table...
                        (mouse,Day,comparison_type,tuning_curves_for_this_comparison,real_ND,shuffled_ND,shuffled_ND_cell_index,...
                        real_distribution_count,shuffled_distribution_count,RankSumP,RankSumH,RankSumSTATS);

                    if exist('ND','var') && sum((strcmp(ND.nd_table.Day,day_cd) + strcmp(ND.nd_table.mouse,mouse_cd))==2)==0
                        nd_table = vertcat(ND.nd_table, nd_table);
                        save(['/home/gaia/Desktop/E144/E144/ND_table_last' num2str(n_trials2compare) 'trials/nd_table_last' num2str(n_trials2compare) 'trials'],'nd_table', '-v7.3')
                    else
                        save(['/home/gaia/Desktop/E144/E144/ND_table_last' num2str(n_trials2compare) 'trials/nd_table_last' num2str(n_trials2compare) 'trials'],'nd_table', '-v7.3')
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