clear all;
load('Y:\sstcre_analysis\fmats\e201\days\day58_Fall.mat', 'Fc3', 'putative_pcs', 'ybinned', ...
'changeRewLoc', 'forwardvel', 'licks')     
% generate tuning curve
%%
bin_size = 3;
track_length = 270;
nbins = track_length/bin_size;
eps = find(changeRewLoc>0);
eps = [eps length(changeRewLoc)];
rewlocs = changeRewLoc(changeRewLoc>0)*(3/2);

% per ep
ep=2;
eprng = eps(ep):eps(ep+1);
ypos = ybinned(eprng);
ypos = ceil(ypos*(3/2)); % gain factor for zahra mice
[time_moving,time_stop] = vr_stop_and_moving_time(ypos);
ypos_mov = ypos(time_moving);
for i = 1:nbins    
    time_in_bin{i} = time_moving(ypos_mov >= (i-1)*bin_size & ypos_mov < i*bin_size);
end 

% make bins via suyash method
pc = putative_pcs{ep};
rewloc = rewlocs(ep);
fc3_ep = Fc3(eprng,:);
moving_cells_activity = fc3_ep(time_moving,pc);
fc3_pc = fc3_ep(:,putative_pcs{ep});
for i = 1:size(moving_cells_activity,2)
    for bin = 1:nbins
        cell_activity(bin,i) = mean(fc3_pc(time_in_bin{bin},i)); 
    end
end 
for bin = 1:nbins
    vel(bin) = mean(forwardvel(time_in_bin{bin}));     
end
cell_activity(isnan(cell_activity)) = 0;


% sort by max value
for c=1:size(cell_activity,2)
    f = cell_activity(:,c);
    if sum(f)>0
        [peakval,peakbin] = max(f);
        peak(c) = peakbin;
    else
        peak(c) = 1;
    end
end

[~,sorted_idx] = sort(peak);
%%
figure;
subplot(2,1,1)
imagesc(normalize(cell_activity(:,sorted_idx))'); 
colormap jet 
xticks([0:90])
xticklabels([0:3:270])
subplot(2,1,2)
plot(vel, 'k'); hold on
rectangle('Position',[(rewloc-5)/3 0 10/3 max(vel)-1], ... % just picked max for visualization
      'FaceColor',[0 .5 .5 0.3])
xticks([0:90])
xticklabels([0:3:270])
