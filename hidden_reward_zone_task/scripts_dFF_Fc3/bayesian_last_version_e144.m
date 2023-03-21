% E144 settings
clear 
clc

day =1;%:10; % [1 2 4 5 6 7 8 9 10];
% Md1 = cell(1,length(day));
for day_num = day
    load('Fall_D11')
    
    %  load Fall_D1
    if exist('rewEpochV2new','var')
        rewEpoch = table2cell(rewEpochV2new);
    elseif exist('rewEpochV2','var')
        rewEpoch = table2cell(rewEpochV2);
    elseif exist('rewEpochV3new','var')
        rewEpoch = table2cell(rewEpochV3new);
    elseif exist('rewEpochV6','var')
        rewEpoch = table2cell(rewEpochV6);
    else
        error('Hey! This data does not contain the rewEpoch structure analyzed with "rewEpochStructure" ')
    end
    % divide the track in bins of X cm, sampled at 32Hz
    nplanes = 1; % number of planes
    Fs = 32/nplanes; % Hz, recording frequency
    X = 2; % cm
    trainTrials = 5; % number of training trials
    rewEp = 1; % epoch of the training
    track_length = 180; %cm
    track = 1:track_length;
    bins = track_length/X; % number of bins
    edges = 0:X:track_length;
    
    cells_all = 1:size(rewEpoch{1, 1}.cell_activity_last ,1); %cells we are working on
    putative_pyrs = intersect(find(skewness(all.dff')>=2),cells_all);
    putative_ints = intersect(find(skewness(all.dff')<2),cells_all);
    cells = putative_pyrs;
    
    binned_track = discretize(track,edges); % here training dtaset
    label = unique(binned_track); % here labels
    
    time_per_bin = histc(binned_track,label); % time the animteal spent in each bin
    p(label) = time_per_bin./sum(time_per_bin); % probability of the animal to be found in a specific bin
    
    dff = cell(1,trainTrials);
    position_raw = cell(1,trainTrials);
    for i = 1:trainTrials
        trial = size(rewEpoch{1,rewEp}.binnedTrial.dFF,2)-i;
        dff{i} = rewEpoch{1,rewEp}.binnedTrial.dFF{1,trial}(cells,:);
        position_raw {i} = rewEpoch{1,rewEp}.binnedTrial.ybinned{1,trial};
    end
    dff = cell2mat(dff);
    position_raw_dff = discretize(cell2mat(position_raw),edges);
    position_raw= cell2mat(position_raw);
    tuning_curves = get_spatial_tuning_all_cells(dff',position_raw',Fs,bins,track_length);
    
    Mdl{day_num} = fitcnb(tuning_curves',label','Distribution', 'kernel');
    % Md2 = fitcnb(dff',position_raw_dff','Distribution', 'kernel');
    
    XTest = all.dff(cells,:)';
    % [label2,score2] = predict(Md2,XTest);
    [label1{day_num},score1{day_num}] = predict(Mdl{day_num},XTest);
    
    clearvars XTest dff position_raw_dff position_raw tuning_curves
end
%
palette = getPyPlot_cMap('gist_heat'); % cubehelix nipy_spectral
day = [1 2 4 5 6 7 8 9 10];
% Md1 = cell(1,length(day));
for day_num = 1
    figure;
    imagesc(score1{day_num}')
    set(gca,'Ydir','normal')
    colormap(palette)
    caxis([0 0.07])
    hold on
    load(['Fall_D' num2str(day_num)])
     y = plot(discretize(ybinned,edges),'Color',[0.6 0.6 0.6 0.4],'LineWidth',2);
    hold off
    title (['Fall_D' num2str(day_num) ' pyrs E144'])
end

