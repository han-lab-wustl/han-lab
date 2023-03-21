%% Tuning curve remapping
% This script plot a figure containing the followiong:
% cell activity + deconvolved spikes
% ypos + fc3
% trial by trial tuning curve
% first 8 and last 8 trials tuing curve
% COM
% COM diff

% It assumes that you run remap_analysis_8trials first, and you save the
% workspace in the same folder of your Fall structure(s).
% initally used Fc3, now it uses spks.

%EB 6/23/21; eleonora.bano@wustl.edu
%**************************************************************************
clc
clear % clear the environment

% Settings ----------------------------------------------------------------
Days = 1; % set the day you want to look@
nplanes = 4;
Fs = 32/nplanes;
nBins = 35;
track_length =180;
n_window = 3;
cells2visualize = '1:numel(cells)'; % which cells are you working with
palette_remap = getPyPlot_cMap('magma'); % choose your palette
name = 'workspace_remap_8trials210624'; % workspac saved after running remap_analysis_8trials 
%--------------------------------------------------------------------------

filename = ['Fall_D' num2str(Days)];
load(filename) % load your file

rewEpoch = table2cell(rewEpochV4); % altough the structure is more readable, working with cells is easier.

original_activity = cell(Days);
rewEpoch_idx = {'1,1','2,2','1,2'}; % indeces of the epochs you are interersted in
for i = 1:numel(rewEpoch_idx)
    % this is to get the sliding COM in 3 trials windows
    rE = eval(['rewEpoch{' rewEpoch_idx{i} '}']);
    for j = 1 : (numel(rE.binnedTrial.trialnum)-n_window+1)
        Fc3 = cell2mat(rE.binnedTrial.spks(j:(j+n_window-1))); % EB 6/24 modified to use spks
        position = cell2mat(rE.binnedTrial.ybinned(j:(j+n_window-1)));
        %         disp([num2str(j:(j+n_window-1)) ';'])
        sliding_cell_activity{days}{i}{j} = ...
            get_spatial_tuning_all_cells(Fc3',position,Fs,nBins,track_length);
        sliding_com {days}{i}{j}= calc_COM(sliding_cell_activity{days}{i}{j});
        clearvars Fc3 position
    end
end
for i = 1:numel(rewEpoch_idx)
    % get single trial tuning curve
    rE = eval(['rewEpoch{' rewEpoch_idx{i} '}']);
    for j = 1 : numel(rE.binnedTrial.trialnum)
        Fc3 = cell2mat(rE.binnedTrial.spks(j));
        position = cell2mat(rE.binnedTrial.ybinned(j));
        original_activity{Days}{i}{j} = ...
            get_spatial_tuning_all_cells(Fc3',position,Fs,nBins,track_length);
        clearvars Fc3 position
    end
end
cell2visualize = eval(cells2visualize);
d_num = Days;

for j = 1 : numel(sliding_cell_activity{d_num})
    % for each epoch, concatenate all the sliding window activity and com
    sameEp{j} = cell2mat(sliding_cell_activity{d_num}{j});
    sameEp_com{j} = cell2mat(sliding_com{d_num}{j});
end
for j = 1 : numel(original_activity{d_num})
    origin_sameEp{j} = cell2mat(original_activity{d_num}{j});
end

activity_mat = cell2mat(sameEp);
com_mat = cell2mat(sameEp_com);
origin_mat = cell2mat(origin_sameEp);% concatenate all different epochs

for i = cell2visualize
    
    single_cell = activity_mat(i,:);
    % sliding window activity of this cell (across this whole recording)
    single_cell_com =  com_mat(i,:);
    % sliding window COM of this cell (across this whole recording)
    single_cell_origin = origin_mat(i,:);
    % tuning curve of this cell of each trial (across this whole recording)
    for j = 1 : length(single_cell)/nBins
        % reshape slinding window activity
        trial_single_cell(j,:) = single_cell((nBins*(j-1)+1):nBins*j);
    end
    for j = 1 : length(single_cell_origin)/nBins
        % reshape tuning curve for each trial
        trial_single_cell_origin(j,:) = single_cell_origin((nBins*(j-1)+1):nBins*j);
    end
    ax1 = figure(d_num*10000+i);
    subplot(2,3,2)
    imagesc(1:10)
    imagesc(trial_single_cell_origin)
    colormap(ax1,palette_remap)
    num_trials = numel(sliding_cell_activity{d_num}{1});
    hold on
    yticks([num_trials+2 num_trials+5]) % ADD Xticks with reward locations
    title('tuning curve by trial')
    
    x = find(~isnan(single_cell_com));
    v = single_cell_com(~isnan(single_cell_com));
    xq = 1:length(single_cell_com);
    d_com = diff(single_cell_com);
    
    subplot(2,3,3)
    plot(single_cell_com,'LineWidth',2)
    title('sliding com')
    
    
    subplot(2,3,6)
    pl2 = plot(d_com,'k','LineWidth',2);
    ylims=get(gca,'ylim');
    legend(pl2,'difference of sliding COM')
    yticks(num_trials)
    box off
    
    linear_cells = 1:numel(cell2visualize);
    index = find(cell2visualize==i);
    cell_random = linear_cells(index);
    real_cell = i;
    %% when do the place field switch from one representation to the other?
    % here single cell visualization
    
    load(name)
    labels = {'First 8 trials rewEp1','Last 8 trials rewEp1','First 8 trials rewEp2' ,'Last 8 trials rewEp2'};
    cmap = flipud(pink(10000));
    filename = ['Fall_D' num2str(d_num)];
    load(filename)
    rewEpoch = table2cell(rewEpochV4);
    
    
    trnum = trialnum+1;
    idx = [0 diff(trnum)==1];
    idxProbes = trnum<4;
    trials = trnum;
    trials(idx==1) = 0;
    trials = bwlabel(trials);
    probes = trials(idxProbes );
    idc = [(diff(probes)~=0) 1];
    probeN = probes(idc==1);
    probeN(probeN==0) = [];
    
    
    hFig = figure(d_num*100000+real_cell);
    for ii = 1: size(cell_r_real,2)
        random_cell = cell_r_shuffled{d_num , ii}(:,cell_random);
        real = cell_r_real{d_num , ii}(cell_random);
        
        subplot(1,size(cell_r_real,2),ii)
        max_n = max(histcounts(random_cell,'BinEdges',(-1:0.2:1)));
        sorted_c = sort(random_cell);
        sorted_c(isnan(sorted_c)) = [];
        if ~isempty(sorted_c)
            indx = numel(sorted_c)*.95;
            cut_off = sorted_c(round(indx));
            p = r_coeff_pVal{d_num,ii}(cell_random);
            p1 = histogram(random_cell, 'FaceColor',cmap(1600,:),'BinEdges',(-1:0.2:1),'FaceAlpha',0.8,'EdgeColor',[1 1 1],'LineWidth',0.2);
            hold on
            p1_2 = histogram(random_cell(random_cell>cut_off), 'FaceColor',cmap(7600,:),'BinEdges',(-1:0.2:1),'FaceAlpha',0.8,'EdgeColor',[1 1 1],'LineWidth',0.2);
            xlim([-1, 1])
            ylim([0,200])
            ylims=get(gca,'ylim');
            xlims=get(gca,'xlim');
            %     text(xlims(1)+0.1,ylims(2)-10,['p = ' num2str(p)])
            hold on
            p2 = plot([real real],[0 ylims(2)],'-r','LineWidth',2);
            if p < 0.05
                text(xlims(1)+0.1,ylims(2)-10,'p < 0.05')
                %     text(real,max_n+1,'\bf *','FontSize',18,'HorizontalAlignment','center')
            end
            xlabel(labels(ii))
            if ii ==1
                ylabel('count of shuffled pearson correlation')
            end
            if ii == size(cell_r_real,2)
                legend([p1 p1_2 p2],'Shuffled r <95%','Shuffled r >= 95%','Original r')
                box off
                figure(d_num*10000+real_cell)
                %             subplot(3,2,4)
                %             max_n = max(histcounts(random_cell,'BinEdges',(-1:0.2:1)));
                %             sorted_c = sort(random_cell);
                %             sorted_c(isnan(sorted_c)) = [];
                %             if ~isempty(sorted_c)
                %             indx = numel(sorted_c)*.95;
                %             cut_off = sorted_c(round(indx));
                %             p = r_coeff_pVal{d_num ,ii}(cell_random);
                %             p1 = histogram(random_cell, 'FaceColor',cmap(1600,:),'BinEdges',(-1:0.2:1),'FaceAlpha',0.8,'EdgeColor',[1 1 1],'LineWidth',0.2);
                %             hold on
                %             p1_2 = histogram(random_cell(random_cell>cut_off), 'FaceColor',cmap(7600,:),'BinEdges',(-1:0.2:1),'FaceAlpha',0.8,'EdgeColor',[1 1 1],'LineWidth',0.2);
                %             xlim([-1, 1])
                %             ylim([0,200])
                %             ylims=get(gca,'ylim');
                %             xlims=get(gca,'xlim');
                %             %     text(xlims(1)+0.1,ylims(2)-10,['p = ' num2str(p)])
                %             hold on
                %             p2 = plot([real real],[0 ylims(2)],'-r','LineWidth',2);
                %             if p < 0.05
                %                 text(xlims(1)+0.1,ylims(2)-10,'p < 0.05')
                %                 %     text(real,max_n+1,'\bf *','FontSize',18,'HorizontalAlignment','center')
                %             end
                %             xlabel(labels(ii))
                %             legend([p1 p1_2 p2],'Shuffled r <95%','Shuffled r >= 95%','Original r')
                %             box off
                
                subplot(2,3,5)
                tc1 = plot(tuning_curve_binned{d_num, 2}(cell_random,:),'LineWidth',2);% last 8 trials ep1
                hold on
                tc2 = plot(tuning_curve_binned{d_num, 4}(cell_random,:),'LineWidth',2);% last 8 trials ep2
                legend([tc1 tc2],'last 8 trials ep1','last 8 trials ep2')
                ylims2=get(gca,'ylim');
                xlims2=get(gca,'xlim');
                text(xlims2(1)+0.1,ylims2(2)-10,['p = ' num2str(p)])
                title('tuning curves')
            end
        end
    end
    mtit(['cell' num2str(cell_random)] )
end
% end
% mtit('Epochs where the same tuning curve of the last three trials of reward Epoch1 is matained')




