function [tuning_curves, coms] = make_tuning_curves(changeRewLoc, trialnum, rewards, ybinned, ...
    licks, forwardvel, thres, Fs, ftol, bin_size, stat, iscell, plns, Fc3)
tuning_curves = {};
coms = {};
for pln=plns
    %     load(fullfile(src, an, 'days', sprintf('%s_day%03d_plane%i_Fall.mat',an, dy, pln)), ...
    %         'dFF', 'Fc3', 'iscell', 'ybinned', 'changeRewLoc', ...
    %             'forwardvel', 'licks', 'trialnum', 'rewards')
    eps = find(changeRewLoc>0);
    eps = [eps length(changeRewLoc)];
    rewlocs = changeRewLoc(changeRewLoc>0)*(gainf);
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
            if trial>=3 && trial>=max(trn)-ntrials % trial < 3, probe trial
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
            for i = 1:nbins
                time_in_bin{i} = time_moving(ypos_mov >= (i-1)*bin_size & ...
                    ypos_mov < i*bin_size);
            end

            % make bins via suyash method
            %     pc = putative_pcs{1};
            pc = logical(iscell(:,1));
            rewloc = rewlocs(ep);
            [~,bordercells] = remove_border_cells_all_cells(stat, Fc3);
            %         moving_cells_activity = fc3_ep(time_moving,:);
            bordercells_pc = bordercells(pc); % mask border cells
            fc3_pc = Fc3(eprng,pc); % only iscell
            fc3_pc = fc3_pc(:,~bordercells_pc); % remove border cells
            % activity binning
            cell_activity = zeros(nbins, size(fc3_pc,2));
            for i = 1:size(fc3_pc,2)
                for bin = 1:nbins
                    cell_activity(bin,i) = mean(fc3_pc(time_in_bin{bin},i));
                end
            end
            cell_activity(isnan(cell_activity)) = 0;

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
        tuning_curves{ep} = cell_activity;
        coms{ep} = calc_COM_EH(cell_activity',bin_size);
    end
end