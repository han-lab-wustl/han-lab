clear all;
% individual day analysis
load('Y:\sstcre_analysis\fmats\e145\days\e145_day011_plane2_Fall.mat', 'dFF', ...
        'Fc3', 'ybinned', 'changeRewLoc', 'forwardvel', 'licks', 'trialnum', 'rewards')  
cc = load('Y:\sstcre_analysis\celltrack\e145_week01-02_plane2\Results\commoncells_atleastoneactivedayperweek_4weeks_week2daymap.mat');
an = 'e145';
dy = 11;
cc = cc.cellmap2dayacrossweeks;
ddt = dy-4;%dy-3; % based on where you started tracking cells from
% generate tuning curve
% save all dffs and fc3 of tracked cells in one struct

bin_size = 3;
track_length = 270;
gainf = 3/2;
nbins = track_length/bin_size;
eps = find(changeRewLoc>0);
eps = [eps length(changeRewLoc)];
rewlocs = changeRewLoc(changeRewLoc>0)*(gainf);
figure('Renderer', 'painters', 'Position', [10 10 1050 800])
tuning_curves = {};
for ep=[1 2 3]
    % per ep    
    eprng = eps(ep):eps(ep+1);
    trn = trialnum(eprng);  
    rew = rewards(eprng)>0.5;
    % opto period
%     mask = trn>=3 & trn<8;
    % no probes
%     mask1 = trn>=8;
    strials = ones(1, length(unique(trn)))*NaN; % only get successful trials
    for trial=unique(trn)
    if trial>=3 % trial < 3, probe trial
        if sum(rew(trn==trial)==1)>0 % if reward was found in the trial
            strials(trial)=trial;        
        end
    end
    end
    strials = strials(~isnan(strials));
    mask = ismember(trn, strials);
    
%     mask = trn<3;
    eprng = eprng(mask);
    
    ypos = ybinned(eprng);
    ypos = ceil(ypos*(gainf)); % gain factor for zahra mice
    [time_moving,time_stop] = vr_stop_and_moving_time(ypos);
    ypos_mov = ypos(time_moving);
    for i = 1:nbins    
        time_in_bin{i} = time_moving(ypos_mov >= (i-1)*bin_size & ypos_mov < i*bin_size);
    end 
    
    % make bins via suyash method
%     pc = putative_pcs{1};
    weeklut = 1:length(cc(:,ddt));
    cellind = cc(:, ddt);    
    pc = cellind(cellind>0);
    weeklutind = weeklut(cellind>0);
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
    
    if ep == 1 % sort by ep 1
%         % sort by max value
%         peak = zeros(1, size(cell_activity,2));
%         for c=1:size(cell_activity,2)
%             f = cell_activity(:,c);
%             if sum(f)>0
%                 [peakval,peakbin] = max(f);
%                 peak(c) = peakbin;
%             else
%                 peak(c) = 1;
%             end
%         end
        % sort by median - ed's code
        com = calc_COM_EH(cell_activity',bin_size);
        
%         [~,sorted_idx] = sort(peak);
        [~,sorted_idx] = sort(com);
    end
    tuning_curves{ep} = cell_activity(:,sorted_idx);
    %%
    
    subplot(1,3,ep)
    imagesc(normalize(tuning_curves{ep},1)'); 
    colormap jet 
    xticks([0:90])
    xticklabels([0:3:270])
    yticks(1:length(weeklutind))
    yticklabels(weeklutind(sorted_idx));
%     subplot(2,1,2)
%     plot(vel, 'k');
%     yyaxis right
%     plot(lk, 'r'); ylim([0 0.5])
%     hold on 
%     rectangle('Position',[(rewloc-5)/3 0 10/3 max(lk)+1], ... % just picked max for visualization
%           'FaceColor',[0 .5 .5 0.3])
%     xticks([0:90])
%     xticklabels([0:3:270])
    title(sprintf('epoch %i', ep))
end
sgtitle(sprintf('animal %s, day %i', an, dy))

tuning_curves_tracked_cells = tuning_curves;
save(fullfile('Y:\sstcre_analysis\fmats\e201\days', ...
sprintf('day%02d_Fall.mat', dy)), 'tuning_curves_tracked_cells', '-append')