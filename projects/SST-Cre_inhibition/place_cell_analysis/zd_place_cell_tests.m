load('Y:\sstcre_analysis\fmats\e201\days\day63_Fall.mat')
eps = find(changeRewLoc);
eps = [eps length(changeRewLoc)]; % includes end of recording as end of a epoch
ep = 1;    
eprng = eps(1):eps(2);
% mask = trialnum(eprng)>=3; % skip probes
ypos = ybinned(eprng);
ypos = ceil(ypos*(3/2)); % gain factor for zahra mice
[time_moving ,time_stop] = vr_stop_and_moving_time(ypos);
fc3 = Fc3(eprng,:)';
% fc3 = fc3(:,mask);
% rewloc = changeRewLoc(changeRewLoc>0); % reward location
% rewlocopto = rewloc(ep)*(3/2);
%%
% check to see if there are any negative transients in fc3
figure; plot(fc3(randi([1 size(fc3,1)],1),:))
xlabel('frames in ep1')
ylabel('fc3')
title(sprintf('cell no . %i', randi([1 size(fc3,1)],1)))
%% 
% calculate spatial info
% only during moving time?
fc3_mov = fc3(:,time_moving);
ypos_mov = ypos(:,time_moving);
bin = 3; % cm t
track_length = 270;
spatial_info = get_spatial_info_all_cells(fc3_mov',ypos_mov,31.25, ...
    ceil(track_length/bin),track_length);

% shuffle params
%shuffle transients and not fc3 values themselves = use bwlabel

for i = 1:size(fc3_mov,1)   
    bins2shuffle_forcell{i} = shuffling_bins(fc3_mov(i,:));   
end 
nshuffles = 5000;
spatial_info_shuf = zeros([nshuffles size(spatial_info,2)]);
shuffledbins_forcell(1:size(fc3_mov,1)) = {0};
for j = 1:nshuffles
    disp(['Shuffle number ', num2str(j)]) 
    parfor i = 1:size(fc3_mov,1)
        s = shuffle(bins2shuffle_forcell{i});
        shuffledbins_forcell{i} = s; 
        sca = fc3_mov(i,cell2mat(shuffledbins_forcell{i}));
        shuffled_cells_activity(i,:) = sca;
    end 
    spatial_info_shuf(j,:) = get_spatial_info_all_cells(shuffled_cells_activity',ypos_mov,31.25, ...
    ceil(track_length/bin),track_length);
end
% compare spatial info to shuffled distribution
putative_pc = zeros(1,size(fc3_mov,1)); % mask for pcs that pass shuffle crtieria
pvals = zeros(1,size(fc3_mov,1)); % mask for pcs that pass shuffle crtieria
for cell=1:size(fc3_mov,1)
    si = spatial_info(cell);
    si_shuf = sum(spatial_info_shuf(:,cell)>si)/nshuffles;
    pvals(cell) = si_shuf;
    if si_shuf<0.01 % 99 cut off
        putative_pc(cell) = 1;
    end
end

% get place cells only
pc_fc3=fc3(logical(putative_pc),:);
%%
% generate tuning curve
bin_size = 1;
nbins = track_length/bin_size;

for i = 1:nbins    
    time_in_bin{i} = time_moving(ypos_mov >= (i-1)*bin_size & ypos_mov < i*bin_size);
end 

moving_cells_activity = fc3(logical(putative_pc),time_moving);
fc3_pc = fc3(logical(putative_pc),:);
for i = 1:size(moving_cells_activity,1)
    for bin = 1:nbins
        cell_activity(i,bin) = mean(fc3_pc(i,time_in_bin{bin})); 
    end
end 
cell_activity(isnan(cell_activity)) = 0;
figure; imagesc(normalize(cell_activity))

