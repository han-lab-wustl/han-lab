% Zahra
% makes tuning curves with velocity filter
% uses suyash's binning method
% per day analysis using iscell boolean
clear all;

% an = 'e200';
% an = 'e201';
% an = 'e145';
an = 'e218';
% an = 'e139';
% individual day analysis
% dys = [1,2,3,5,6,7,8,9,10];
dys = [20,21,22,24,25];
% dys = [50:52,54]%:73, 75];
% dys = [62:67,69:70,72:74,76,81:85];
% dys = [4:7, 9:11];
savedst = 'Y:\sstcre_analysis\figures'; % for figures
src = 'X:\vipcre'; % folder where fall is 
% src = 'Y:\sstcre_analysis\fmats';
for dy=dys
    clearvars -except dys an cc dy src savedst
    pth = dir(fullfile(src, an, string(dy), '**\*Fall.mat'));
    % load vars
    load(fullfile(pth.folder,pth.name), 'dFF', ...
            'Fc3', 'iscell', 'ybinned', 'changeRewLoc', ...
            'forwardvel', 'licks', 'trialnum', 'rewards')              
      % for copied falls
    plns = [0];
    for pln=plns
%     load(fullfile(src, an, 'days', sprintf('%s_day%03d_plane%i_Fall.mat',an, dy, pln)), ...
%         'dFF', 'Fc3', 'iscell', 'ybinned', 'changeRewLoc', ...
%             'forwardvel', 'licks', 'trialnum', 'rewards')              
    % vars to get com and tuning curves
    bin_size = 3;
    gainf = 3/2; % 3/2 VS. 1
    track_length = 180*gainf; % 270, 180    
    rew_zone = 10; % cm
    nbins = track_length/bin_size;
    eps = find(changeRewLoc>0);
    eps = [eps length(changeRewLoc)];
    rewlocs = changeRewLoc(changeRewLoc>0)*(gainf);    
    tuning_curves = {};
    for ep=1:length(eps)-1
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
        % todo: may need a condition that get last few trials only, etc.
        % && trial>=max(trn)-8 <- last 8 trials
        if trial>=3 && trial>=max(trn)-8 % trial < 3, probe trial
            if sum(rew(trn==trial)==1)>0 % if reward was found in the trial
                strials(trial)=trial;        
            end
        end
        end
        strials = strials(~isnan(strials)); % only uses successful trials
        mask = ismember(trn, strials);
        
    %     mask = trn<3;
        eprng = eprng(mask);
        if ~isempty(eprng)
        ypos = ybinned(eprng);
        ypos = ceil(ypos*(gainf)); % gain factor for zahra mice
        [time_moving,time_stop] = vr_stop_and_moving_time(ypos);
        ypos_mov = ypos(time_moving);
        for i = 1:nbins    
            time_in_bin{i} = time_moving(ypos_mov >= (i-1)*bin_size & ypos_mov < i*bin_size);
        end 
        
        % make bins via suyash method
    %     pc = putative_pcs{1};
        pc = logical(iscell(:,1));
        rewloc = rewlocs(ep);
        fc3_ep = Fc3(eprng,:);
        moving_cells_activity = fc3_ep(time_moving,:);
        fc3_pc = fc3_ep(:,pc);
        % mean length of transients         
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
        end
        % overwrite ep with previous ep if eprng does not exist
        tuning_curves{ep} = cell_activity;  
        coms{ep} = calc_COM_EH(cell_activity',bin_size);
        
    end

    % if less than 3 ep
    if length(eps)> 3
        comparisons = {[1 2], [1 3], [2 3]};
    else
        comparisons = {[1 2]};
    end
    for i=1:length(comparisons)
        comparison = comparisons{i};
        [p,h,stat] = do_tuning_curve_ranksum_test(tuning_curves{comparison(1)}', ...
            tuning_curves{comparison(2)}');
        pvals(i) = p;
        disp(p)
        slideId = pptx.addSlide();
        fprintf('Added slide %d\n',slideId);
        fig = figure('Renderer', 'painters');
        plot(coms{comparison(1)}, coms{comparison(2)}, 'ko'); hold on;
        xline(rewlocs(comparison(1)), 'r', 'LineWidth', 3);
        yline(rewlocs(comparison(2)), 'r', 'LineWidth', 3)
        plot([0:track_length],[0:track_length], 'k', 'LineWidth',2)
        xlim([0 track_length]); ylim([0 track_length])
        xlabel(sprintf('ep%i', comparison(1)));
        ylabel((sprintf('ep%i', comparison(2))))
        title(sprintf(['COM (median) \n ' ...
            'animal %s, day %i, plane %i \n ' ...
            'comparison: ep%i vs ep%i'], an, dy, pln, comparison(1), comparison(2)))
        close(fig)
    end
    fig = figure('Renderer', 'painters', 'Position', [10 10 1050 800]);
    for ep=1:length(eps)-1 
        % only analyse until ep 3
        if ep<=3
            subplot(1,3,ep)
            plt = tuning_curves{ep}';
            imagesc(normalize(plt(sorted_idx,:),2)); 
            hold on;
            % plot rectangle of rew loc
            % everything divided by 3 (bins of 3cm)
            rectangle('position',[ceil(rewlocs(ep)/bin_size)-ceil((rew_zone/bin_size)/2) 0 ...
            rew_zone/bin_size size(cell_activity,2)], ... % just picked max for visualization
                        'EdgeColor',[0 0 0 0],'FaceColor',[1 1 1 0.5])
            colormap jet 
            xticks([0:bin_size:ceil(track_length/bin_size)])
            xticklabels([0:bin_size*bin_size:track_length])
            title(sprintf('epoch %i', ep))
        end
    end
    
    sgtitle(sprintf(['animal %s, day %i, plane %i \n ' ...
            'ep1 vs ep2 = %d \n ep1 vs ep3 = %d \n ep2 vs ep3 = %d'], an, dy, pln, ...
            pvals))
%     savefig(fullfile(savedst,sprintf('%s_day%i_tuning_curves_w_ranksum.fig',an,dy)))
    close(fig)
    end
end