%E144
clear
clc

cell_type = 'ints'; % pyrs or ints
day =1; %[1 2 4 5 6 7 8 9 10];
% Md1 = cell(1,length(day));
for day_num = day
 load(['Fall_D' num2str(day_num)])
 
%  load Fall_D1
if exist('rewEpochV2new','var')
rewEpoch = table2cell(rewEpochV2new);
elseif exist('rewEpochV2','var')
rewEpoch = table2cell(rewEpochV2);
elseif exist('rewEpochV3new','var')
rewEpoch = table2cell(rewEpochV3new);
elseif exist('rewEpochV4','var')
rewEpoch = table2cell(rewEpochV4);
elseif ~exist('rewEpoch','var')
    error('Hey! This data does not contain the rewEpoch structure analyzed with "rewEpochStructure" ')
end
% divide the track in bins of X cm, sampled at 32Hz
nplanes = 1; % number of planes
Fs = 32/nplanes; %Hz, recording frequency
X = 2; %cm
trainTrials = 5; % number of training trials
rewEp = 1; % epoch of the training
track_length = 180; %cm
bins = track_length/X; % number of bins
edges = 0:X*2/180:2;
[~, rewLocs] = get_reward_location_Idxs(changeRewLoc);
rewL = rewLocs(rewEp);


cells_all = 1:size(rewEpoch{1, 1}.cell_activity ,1); %cells we are working on
pyrs = intersect(find(skewness(all.dff')>=2),cells_all); % only pyramidal cells
ints = intersect(find(skewness(all.dff')<2),cells_all); % only ints cells
cells = eval(cell_type);
position_real = 1:180;
track = relative2rew(position_real,rewL);
[binned_Track,Edges] = discretize(track,edges); % here training dtaset
binned_track = Edges(binned_Track);
label = unique(binned_track); % here labels

time_per_bin = histc(binned_track,label); % time the animal spent in each bin    

dff = cell(1,trainTrials);
position_raw = cell(1,trainTrials);
stop_bins = cell(1,trainTrials);
for i = 1:trainTrials
trial = size(rewEpoch{1,rewEp}.binnedTrial.dFF,2)-i;
dff{i} = rewEpoch{1,rewEp}.binnedTrial.dFF{1,trial}(cells,:);
position_raw {i} = relative2rew(rewEpoch{1,rewEp}.binnedTrial.ybinned{1,trial},rewLocs(rewEp));
stop_bins {i} = [0 diff(rewEpoch{1,rewEp}.binnedTrial.ybinned{1,trial})<5/Fs]; 
end
stop_raw = cell2mat(stop_bins);
dff = cell2mat(dff);
dff(:,stop_raw==1) = [];
position_raw = cell2mat(position_raw);
position_raw(stop_raw==1) = [];
position_raw_dff = discretize(position_raw,edges);
tuning_curves = get_spatial_tuning_all_cells_decoderRewCenter(dff',position_raw',Fs,bins,2);
label = 1:size(tuning_curves,2);
Mdl{day_num} = fitcnb(tuning_curves',label','Distribution', 'kernel');
XTest = all.dff(cells,:)';
% [label2,score2] = predict(Md2,XTest);
[label1{day_num},score1{day_num}] = predict(Mdl{day_num},XTest);

clearvars XTest dff position_raw_dff position_raw tuning_curves
end
%
palette = getPyPlot_cMap('gist_heat'); % cubehelix nipy_spectral
% Md1 = cell(1,length(day));
for day_num = day
figure;
imagesc(score1{day_num}')
set(gca,'Ydir','normal')
colormap(palette)
caxis([0 0.07])
hold on
load(['Fall_D' num2str(day_num)])
    change_rewLoc = changeRewLoc>0;
    indeces3 = [0 diff(trialnum==3)];
    idxs_probes = find(indeces3==1);
    idxs_rl = find(changeRewLoc>0);
    if idxs_rl(2)>idxs_probes(1)
        ind = [1 idxs_probes(2:end) length(change_rewLoc)];
    else
        ind = [1 idxs_probes length(change_rewLoc)];
    end
 [~, rewLocs] = get_reward_location_Idxs(changeRewLoc);
for i = 1 : numel(ind)-1
rewLoc = rewLocs(i);
position_real = ybinned(ind(i):ind(i+1));
relative2rew_pos{i} = relative2rew(position_real,rewLoc);
end
ypos = rescale(cell2mat(relative2rew_pos),1,90);
 y = plot(ypos,'Color',[0.6 0.6 0.6 0.2],'LineWidth',1);
hold off
title (['Fall_D' num2str(day_num) cell_type ' E144'])

clearvars relative2rew_pos ypos
end

