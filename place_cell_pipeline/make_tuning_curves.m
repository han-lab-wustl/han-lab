function [tuning_curves, coms, median_com, peak] = make_tuning_curves(eps, trialnum, rewards, ybinned, gainf, ntrials,...
    licks, forwardvel, thres, Fs, ftol, bin_size, track_length, fc3, dff)
nbins = track_length/bin_size;
% TODO: incorporate multi planes
% for pln=plns
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
        ypos = ceil(ypos*(gainf));
        lick = licks(eprng);
        fv = forwardvel(eprng);
        % updated how we get moving time to be consistent with dop pipeline
        [time_moving,~] = get_moving_time_V3(fv, thres, Fs, ftol);
        ypos_mov = ypos(time_moving);
        % ypos_mov(ypos_mov<=3)=8;
        for i = 1:nbins
            time_in_bin{i} = time_moving(ypos_mov >= (i-1)*bin_size & ...
                ypos_mov < i*bin_size);
        end

        % make bins via suyash method
        %     pc = putative_pcs{1};
        % smooth
        fc3_pc = fc3(eprng,:); 
        dff_pc = dff(eprng, :);
        % fc3_pc = smoothdata(fc3_pc, 'gaussian', 3);
        % activity binning
        cell_activity = zeros(nbins, size(fc3_pc,2));
        % to get accurate coms
        cell_activity_dff = zeros(nbins, size(fc3_pc,2));
        for i = 1:size(fc3_pc,2)
            for bin = 1:nbins
                cell_activity(bin,i) = mean(fc3_pc(time_in_bin{bin},i));
                cell_activity_dff(bin,i) = mean(dff_pc(time_in_bin{bin},i));
            end
        end
        cell_activity(isnan(cell_activity)) = 0;
        cell_activity_dff(isnan(cell_activity_dff)) = 0;

        %             if ep == 1 % sort by ep 1
        %                 %         % sort by max value
        %                 %         peak = zeros(1, size(cell_activity,2));
        %                 %         for c=1:size(cell_activity,2)
        %                 %             f = cell_activity(:,c);
        %                 %             if sum(f)>0
        %                 %                 [peakval,peakbin] = max(f);
        %                 %                 peak(c) = peakbin;
        %                 %             else
        %                 %                 peak(c) = 1;
        %                 %             end
        %                 %         end
        %                 % sort by median - ed's code
        %                 com = calc_COM_EH(cell_activity',bin_size);
        %
        %                 %         [~,sorted_idx] = sort(peak);
        %                 [~,sorted_idx] = sort(com);
        %             end
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