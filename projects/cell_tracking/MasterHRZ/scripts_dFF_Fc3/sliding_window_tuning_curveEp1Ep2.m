% sliding window tuning curve analysis

clc
clear
days = 1;

filename = ['Fall_D' num2str(days)];

load(filename)
rewEpoch = table2cell(rewEpochV4);
%     if size(rewEpoch,2)>=2 && numel(rewEpoch{1, 2}.binnedTrial.trialnum)> 9   % we need at least two rew epochs completed

Fs = 8;
nBins = 35;
track_length =180;
n_window = 3;

rewEpoch_idx = {'1,1','2,2','1,2'} ;
for i = 1:numel(rewEpoch_idx)
    rE = eval(['rewEpoch{' rewEpoch_idx{i} '}']);
    for j = 1 : (numel(rE.binnedTrial.trialnum)-n_window+1)
        Fc3 = cell2mat(rE.binnedTrial.Fc3(j:(j+n_window-1)));
        position = cell2mat(rE.binnedTrial.ybinned(j:(j+n_window-1)));
        %         disp([num2str(j:(j+n_window-1)) ';'])
        sliding_cell_activity{days}{i}{j} = ...
            get_spatial_tuning_all_cells(Fc3',position,Fs,nBins,track_length);
        sliding_com {days}{i}{j}= calc_COM(sliding_cell_activity{days}{i}{j});
        clearvars Fc3 position
    end
end
for i = 1:numel(rewEpoch_idx)
    rE = eval(['rewEpoch{' rewEpoch_idx{i} '}']);
    for j = 1 : numel(rE.binnedTrial.trialnum)
        Fc3 = cell2mat(rE.binnedTrial.Fc3(j));
        position = cell2mat(rE.binnedTrial.ybinned(j));
        %         disp([num2str(j:(j+n_window-1)) ';'])
        original_activity{days}{i}{j} = ...
            get_spatial_tuning_all_cells(Fc3',position,Fs,nBins,track_length);
        clearvars Fc3 position
    end
end


cell2visualize = rewEpoch{1, 1}.spatial_info_place_cells;
% days = 1;
d_num = days;
palette_remap = getPyPlot_cMap('magma'); % cubehelix nipy_spectral
%subplot(2,2,1)
for i = cell2visualize
    for j = 1 : numel(sliding_cell_activity{d_num})
        sameEp{j} = cell2mat(sliding_cell_activity{d_num}{j});
        sameEp_com{j} = cell2mat(sliding_com{d_num}{j});
    end
    for j = 1 : numel(original_activity{d_num})
        origin_sameEp{j} = cell2mat(original_activity{d_num}{j});
    end
    activity_mat = cell2mat(sameEp);
    com_mat = cell2mat(sameEp_com);
    origin_mat = cell2mat(origin_sameEp);
    
    single_cell = activity_mat(i,:);
    single_cell_com =  com_mat(i,:);
    single_cell_origin = origin_mat(i,:);
    for j = 1 : length(single_cell)/nBins
        trial_single_cell(j,:) = single_cell((nBins*(j-1)+1):nBins*j);
    end
    for j = 1 : length(single_cell_origin)/nBins
        trial_single_cell_origin(j,:) = single_cell_origin((nBins*(j-1)+1):nBins*j);
    end
    ax1 = figure(d_num*10000+i);
    subplot(3,2,1)
    imagesc(1:10)
    imagesc(constrain_row(trial_single_cell))
    colormap(ax1,palette_remap)
    num_trials = numel(sliding_cell_activity{d_num}{1});
    hold on
%     line([1 35],[num_trials num_trials],'Color','white','LineStyle','--')
    yticks(num_trials)
    title('sliding window tuning curve')
    
    subplot(3,2,2)
    imagesc(1:10)
    imagesc(constrain_row(trial_single_cell_origin))
    colormap(ax1,palette_remap)
    num_trials = numel(sliding_cell_activity{d_num}{1});
    hold on
    yticks([num_trials+2 num_trials+5])
%      num_trials = num_trials+2;
%     line([1 35],[num_trials num_trials],'Color','white','LineStyle','--')
%     num_trials = num_trials+3;
%     line([1 35],[num_trials num_trials],'Color','white','LineStyle','--')
    title('trial by trial tuning curve')
    
    % single_cell_com(isnan(single_cell_com)) = [];
    x = find(~isnan(single_cell_com));
    v = single_cell_com(~isnan(single_cell_com));
    xq = 1:length(single_cell_com);
    single_cell_com_interp = interp1(x,v,xq);
    
    d_com = diff(single_cell_com);
    d_com_interp = diff(single_cell_com_interp);
    
    subplot(3,2,3)
    pl1 = plot(d_com_interp,'r','LineWidth',1);
    hold on
    pl2 = plot(d_com,'k','LineWidth',2);
    ylims=get(gca,'ylim');
    legend([pl1 pl2],'difference of interpolated COM','difference of real COM')
    box off
    % line([num_trials-3 num_trials-3],ylims,'Color',[0.6 0.6 0.6],'LineStyle','--','LineWidth',2)
    
    subplot(3,2,5)
    plot(single_cell_com,'LineWidth',2)
    title('sliding com')
    
    linear_cells = 1:numel(cell2visualize);
    index = find(cell2visualize==i);
    cell_random = linear_cells(index);
    real_cell = i;
    %% when do the place field switch from one representation to the other?
    % here single cell visualization
    
    load('workspace_remap_8trials')
    labels = {'First 8 trials rewEp1','Last 8 trials rewEp1','First 8 trials rewEp2' ,'Last 8 trials rewEp2'};
    cell_random = cell_random;
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
    
    
    hFig = figure;
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
            figure(10000+real_cell)
            subplot(3,2,4)
            max_n = max(histcounts(random_cell,'BinEdges',(-1:0.2:1)));
            sorted_c = sort(random_cell);
            sorted_c(isnan(sorted_c)) = [];
            if ~isempty(sorted_c)
            indx = numel(sorted_c)*.95;
            cut_off = sorted_c(round(indx));
            p = r_coeff_pVal{d_num ,ii}(cell_random);
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
            legend([p1 p1_2 p2],'Shuffled r <95%','Shuffled r >= 95%','Original r')
            box off
            
            subplot(3,2,6)
            tc1 = plot(tuning_curve_binned{d_num, 2}(cell_random,:),'LineWidth',2);% last 8 trials ep1
            hold on
            tc2 = plot(tuning_curve_binned{d_num, 4}(cell_random,:),'LineWidth',2);% last 8 trials ep2
            legend([tc1 tc2],'last 8 trials ep1','last 8 trials ep2')
            title('tuning curves')
            end
            end
        end
    end
    % end
    % mtit('Epochs where the same tuning curve of the last three trials of reward Epoch1 is matained')
    mtit(['cell' num2str(cell_random)] )
end