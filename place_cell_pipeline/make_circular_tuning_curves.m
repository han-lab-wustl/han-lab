function [tuning_curves, coms, median_com, peak] = make_circular_tuning_curves(eps, trialnum, rewards, ybinned, gainf, ntrials,...
    licks, forwardvel, thres, Fs, ftol, bin_size, fc3, dff, nbins)
% TODO: incorporate multi planes
% for pln=plns
% w/o velocity filter
tuning_curves = {}; coms = {};
for ep=1:length(eps)-1
    % per ep
    eprng = eps(ep):eps(ep+1);
    trn = trialnum(eprng);
    rew = rewards(eprng)>0.5;
    % opto period
    % trn>=3 & trn<8;
    % no probes
    % mask1 = trn>=8;
    strials = ones(1, length(unique(trn)))*NaN; % only get successful trials
    for trial=unique(trn)
        if trial>=3 && trial>=max(trn)-ntrials%trial>min(trn)+ntrials%trial>=max(trn)-ntrials % trial < 3, probe trial
            %                 if sum(rew(trn==trial)==1)>0 % if reward was found in the trial
            %                     strials(trial)=trial;
            %                 end
            strials(trial)=trial; % successful and fail trials
        end
    end
    strials = strials(~isnan(strials)); % only uses successful trials
    mask = ismember(trn, strials);

    eprng = eprng(mask);
    if ~isempty(eprng)
        ypos = ybinned(eprng);
        % ypos = ceil(ypos*(gainf));
        lick = licks(eprng);
        fv = forwardvel(eprng);
        [time_moving,~] = get_moving_time_V3(fv, thres, Fs, ftol);
        ypos_mov = ypos(time_moving);
        % Initialize cell array for binned time
        time_in_bin = cell(nbins, 1);
        
        for i = 1:nbins
            bin_indices = find(ypos_mov >= (i-1) * bin_size & ypos_mov < i * bin_size);    
            % Store time points in the corresponding bin
            time_in_bin{i} = time_moving(bin_indices);
        end

        % make bins via suyash method
        fc3_pc = fc3(eprng,:); 
        dff_pc = dff(eprng, :);
        % activity binning
        cell_activity = zeros(nbins, size(fc3_pc,2));
        % to get accurate coms
        cell_activity_dff = zeros(nbins, size(fc3_pc,2));
        for i = 1:size(fc3_pc,2)
            for bin = 1:nbins                
                cell_activity(bin,i) = mean(fc3_pc(time_in_bin{bin},i), 'omitnan');
                cell_activity_dff(bin,i) = mean(dff_pc(time_in_bin{bin},i), 'omitnan');
            end
        end
        cell_activity(isnan(cell_activity)) = 0;
        cell_activity_dff(isnan(cell_activity_dff)) = 0;

    end
    % overwrite ep with previous ep if eprng does not exist
    tuning_curves{ep} = cell_activity';
    median_com = calc_COM_EH(cell_activity',bin_size);
    % sort by max value
    peak = zeros(1, size(cell_activity,2));
    for c=1:size(cell_activity,2)
    f = cell_activity(:,c);
    if sum(f)>0
        [peakval,peakbin] = max(f);
        peak(c) = peakbin*bin_size;
    else
        peak(c) = 0;
    end
    end
    coms{ep} = median_com;
end
% end
end