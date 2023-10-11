clear all;
dy = 55;
an = 'e201';
load(fullfile('Y:\sstcre_analysis\fmats\e201\days', sprintf('day%02d_Fall.mat', dy)), 'dFF', ...
    'Fc3', 'putative_pcs', 'ybinned', 'changeRewLoc', 'forwardvel', 'licks', 'trialnum')     
% generate tuning curve
%%
bin_size = 3;
track_length = 270;
nbins = track_length/bin_size;
eps = find(changeRewLoc>0);
eps = [eps length(changeRewLoc)];
rewlocs = changeRewLoc(changeRewLoc>0)*(3/2);
% opto period
mask = trn>=3 & trn<8;
% no probes
% mask = trn>=8;
for ep=[1 2 3]
    % per ep    
    eprng = eps(ep):eps(ep+1);
    trn = trialnum(eprng);    
    eprng = eprng(mask);
    ypos = ybinned(eprng);
    ypos = ceil(ypos*(3/2)); % gain factor for zahra mice
    [time_moving,time_stop] = vr_stop_and_moving_time(ypos);
    ypos_mov = ypos(time_moving);
    for i = 1:nbins    
        time_in_bin{i} = time_moving(ypos_mov >= (i-1)*bin_size & ypos_mov < i*bin_size);
    end 
    
    % make bins via suyash method
    pc = putative_pcs{2};
    rewloc = rewlocs(ep);
    fc3_ep = Fc3(eprng,:);
    moving_cells_activity = fc3_ep(time_moving,pc);
    fc3_pc = fc3_ep(:,pc);
    % mean length of transients
    tr_len = zeros(1,size(fc3_pc,2));
    for c=1:size(fc3_pc,2)
        cell_fc3 = fc3_pc(:,c);
        diffs = diff(cell_fc3>0);        
        starts = find(diffs==1);
        stops = find(diffs==-1);
        if size(starts,1)>size(stops,1) % to account for parts of transients being cut off from trialnum chunkin
            stops(end+1) = size(cell_fc3,1);                
        end
        starts_ = starts;
        if size(starts,1)<size(stops,1)
            starts_ = zeros(size(stops,1),1);
            starts_(2:end) = starts;
            starts_(1) = 0;
        end       
        if size(stops,2)==size(starts_,1)
            stops = stops';
        end
        start_stop = stops-starts_;
        
        tr_len(c) = mean(start_stop, 'omitnan');
    end
    tr_len = tr_len(tr_len>0); % removes negative lengths for now
    % may want to deal with this later as it it basically because of
    % lingering transients in the beginning and end of time period
    % maybe smaller in 
    cell_activity = zeros(nbins, size(fc3_pc,2));
    for i = 1:size(fc3_pc,2)
        for bin = 1:nbins
            cell_activity(bin,i) = mean(fc3_pc(time_in_bin{bin},i)); 
        end
    end 
    lick = licks(eprng);
    fv = forwardvel(eprng);
    for bin = 1:nbins
        vel(bin) = mean(fv(time_in_bin{bin}));     
        lk(bin) = mean(lick(time_in_bin{bin}));     
    end
    cell_activity(isnan(cell_activity)) = 0;
    
    % sort by max value
    peak = zeros(1, size(cell_activity,2));
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
    plot(vel, 'k');
    yyaxis right
    plot(lk, 'r'); ylim([0 0.5])
    hold on 
    rectangle('Position',[(rewloc-5)/3 0 10/3 max(lk)+1], ... % just picked max for visualization
          'FaceColor',[0 .5 .5 0.3])
    xticks([0:90])
    xticklabels([0:3:270])
    sgtitle(sprintf('animal %s, day %i, epoch %i', an, dy, ep))
end