% Zahra - Nov 2023
% makes tuning curves with velocity filter
% uses suyash's binning method

% per day analysis using iscell boolean and putative place cells identified
% from spatial info shuffle

% calls functions to calc dff, fc3, putative place cells, 
% make tuning curves, etc. uses median com

% this run script mostly makes plots but calls other functions
% add han-lab and han-lab-archive repos to path!
clear all; 

an = 'e216';
% individual day analysis 
% dys = [27:30, 32:3 4,36,38,40:75];
dys = [55];%[37:42];%[33,35:42];
% dys = [4:7,9:11];
% dys = [1:51];
src = 'X:\vipcre'; % folder where fall is
savedst = 'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\figure_data'; % where to save ppt of figures
% src = 'Y:\sstcre_analysis\fmats';
% pptx    = exportToPPTX(fullfile(savedst,sprintf('%s_tuning_curves_w_ranksum_opto',an)));
pptx    = exportToPPTX('', ... % make new file
    'Dimensions',[12 6], ...
    'Title','tuning curves', ...
    'Author','zahra', ...
    'Subject','Automatically generated PPTX file', ...
    'Comments','This file has been automatically generated by exportToPPTX');

for dy=dys % for loop per day
    clearvars -except dys an cc dy src savedst pptx
    pth = dir(fullfile(src, an, string(dy), '**\*Fall.mat'));
%     pth = dir(fullfile(src, an, 'days', sprintf('*_day%03d*plane0*', dy)));
    % load vars
    load(fullfile(pth.folder,pth.name), 'dFF', ...
        'Fc3', 'stat', 'iscell', 'ybinned', 'changeRewLoc', ...
        'forwardvel', 'licks', 'trialnum', 'rewards', 'tuning_curves', 'coms', ...
        'putative_pcs', 'VR')
    % vars to get com and tuning curves
    bin_size = 3; % cm
    try
        gainf = 1/VR.scalingFACTOR;
    catch
        gainf = 3/2; % 3/2 VS. 1; in this pipeline the gain is multiplied everywhere
    end
    track_length = 180*gainf;
    try
        rew_zone = VR.settings.rewardZone*gainf; % cm
    catch
        rew_zone = 15;
    end
    % zahra hard coded to be consistent with the dopamine pipeline
    thres = 5; % 5 cm/s is the velocity filter, only get
    % frames when the animal is moving faster than that
    ftol = 10; % number of frames length minimum to be considered stopped
    ntrials = 8; % e.g. last 8 trials to compare    
    plns = [0]; % number of planes
    Fs = 31.25/length(plns);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CHECKS %%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%
    if exist('dFF', 'var')==1
    else % make dff and fc3
        fprintf('********calculating dFF and Fc3********\n')
        [~, dFF, Fc3] = create_dff_fc3(fullfile(pth.folder,pth.name), Fs);
        fprintf('********made dFF and Fc3 since they did not exist in structure********')
    end
    % check to see if changerewloc same length as fc3 (only a problem for old
    % multiplane rec
    if (size(Fc3,1)<size(changeRewLoc,2))
        ybinned = ybinned(1:end-1);
        changeRewLoc = changeRewLoc(1:end-1);
        forwardvel = forwardvel(1:end-1);
        trialnum = trialnum(1:end-1);
        rewards = rewards(1:end-1);
    end
    eps = find(changeRewLoc>0);
    eps = [eps length(changeRewLoc)];    
    rewlocs = changeRewLoc(changeRewLoc>0)*(gainf);
    rewzonenum = get_rewzones(rewlocs, gainf); % get rew zone identity too:  a=[{67:86} {101:120} {135:154}];
    if exist('putative_pcs', 'var')==1
    else % run place cell shuffle on only iscell and excludes bordercells
        fprintf('******** \n calculating place cells based on spatial  info shuffle, \n this may take a while...\n********')
        putative_pcs = get_place_cells_all_ep(stat, Fc3, iscell, ...
            changeRewLoc, ybinned,forwardvel,bin_size,track_length,gainf,Fs, ...
            fullfile(pth.folder,pth.name));
        fprintf('********got place cells based on spatial info shuffle********\n')
    end
%     if exist('tuning_curves','var') == 1 && exist('coms','var') == 1 % check if struct already has these saved
%     else
        [tuning_curves, coms] = make_tuning_curves(eps, changeRewLoc, trialnum, rewards, ...
            ybinned, gainf, ntrials,... # makes tuning curves based on last n trials (successful only)
            licks, forwardvel, thres, Fs, ftol, bin_size, track_length, stat, ...
            iscell, plns, Fc3, putative_pcs);
        fprintf('********calculated tuning curves!********\n')
%     end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END OF CHECKS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % do per ep comparison
    comparisons = nchoosek(1:sum(cellfun(@(x) ~isempty(x),tuning_curves)),2);
    rewloccomp = zeros(size(comparisons,1),2); rewzonecomp = zeros(size(comparisons,1),2);
    for i=1:size(comparisons,1)
        comparison = comparisons(i,:);
        if exist('ep_comp_pval', 'var') == 1 % if pvals already calculated
            pvals  = ep_comp_pval(:,3);
            p = pvals(i);
        else
            [p,h,s] = do_tuning_curve_ranksum_test(tuning_curves{comparison(1)}', ...
                tuning_curves{comparison(2)}');
            pvals(i) = p;
        end
        disp(p)        
        slideId = pptx.addSlide();
        fprintf('Added slide %d\n',slideId);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%fig 1%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        fig = figure('Renderer', 'painters');
        subplot(1,2,1)
        plt = tuning_curves{comparison(1)}';
        [~,sorted_idx] = sort(coms{comparison(1)}); % sorts first tuning curve rel to another
        imagesc(normalize(plt(sorted_idx,:),2));
        % plot rectangle of rew loc
        % everything divided by 3 (bins of 3cm)
        rectangle('position',[ceil(rewlocs(comparison(1))/bin_size)-ceil((rew_zone/bin_size)/2) 0 ...
            rew_zone/bin_size size(plt,1)], ... 
            'EdgeColor',[0 0 0 0],'FaceColor',[1 1 1 0.5])
        colormap jet
        xticks([0:bin_size:ceil(track_length/bin_size)])
        xticklabels([0:bin_size*bin_size:track_length])
        title(sprintf('epoch %i', comparison(1)))
        hold on;
        subplot(1,2,2)
        plt = tuning_curves{comparison(2)}';
        imagesc(normalize(plt(sorted_idx,:),2));
        % plot rectangle of rew loc
        % everything divided by 3 (bins of 3cm)
        rectangle('position',[ceil(rewlocs(comparison(2))/bin_size)-ceil((rew_zone/bin_size)/2) 0 ...
            rew_zone/bin_size size(plt,1)], ... 
            'EdgeColor',[0 0 0 0],'FaceColor',[1 1 1 0.5])
        colormap jet
        xticks([0:bin_size:ceil(track_length/bin_size)])
        xticklabels([0:bin_size*bin_size:track_length])
        title(sprintf('epoch %i', comparison(2)))
        sgtitle(sprintf(['animal %s, day %i \n' ...
            'ep%i vs ep%i: ranksum = %d'], an, dy, comparison(1), comparison(2),...
            p))
        pptx.addPicture(fig);
        close(fig)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%fig 2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
            'animal %s, day %i,' ...
            'comparison: ep%i vs ep%i'], an, dy, comparison(1), comparison(2)))
        pptx.addPicture(fig);
        close(fig)
        rewloccomp(i,:) = [rewlocs(comparison(1)) rewlocs(comparison(2))]';
        rewzonecomp(i,:) = [rewzonenum(comparison(1)) rewzonenum(comparison(2))]';
    end    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%fig 3%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    slideId = pptx.addSlide();
    fprintf('Added slide %d\n',slideId);
    fig = figure('Renderer', 'painters', 'Position', [10 10 1050 800]);
    for ep=1:length(eps)-1
        subplot(1,length(eps)-1,ep)
        plt = tuning_curves{ep}';
        % sort all by ep 1
        [~,sorted_idx] = sort(coms{1});
        imagesc(normalize(plt(sorted_idx,:),2));
        hold on;
        % plot rectangle of rew loc
        % everything divided by 3 (bins of 3cm)
        rectangle('position',[ceil(rewlocs(ep)/bin_size)-ceil((rew_zone/bin_size)/2) 0 ...
            rew_zone/bin_size size(plt,1)], ... 
            'EdgeColor',[0 0 0 0],'FaceColor',[1 1 1 0.5])
        colormap jet
        xticks([0:bin_size:ceil(track_length/bin_size)])
        xticklabels([0:bin_size*bin_size:track_length])
        title(sprintf('epoch %i', ep))
    end

    sgtitle(sprintf(['animal %s, day %i'], an, dy))
    %     savefig(fullfile(savedst,sprintf('%s_day%i_tuning_curves_w_ranksum.fig',an,dy)))
    pptx.addPicture(fig);        

    % also append fall with tables    
    ep_comp_pval = array2table([comparisons pvals' rewloccomp rewzonecomp], ...
        'VariableNames', {'ep_comparison1', 'ep_comparison2', 'cs_ranksum_pval', 'rewloc1', ...
        'rewloc2', 'rewzone_ep1', 'rewzone_ep2'});
    save(fullfile(pth.folder,pth.name), 'ep_comp_pval', 'coms','tuning_curves', '-append')
end

% save ppt
fl = pptx.save(fullfile(savedst,sprintf('%s_tuning_curves_w_ranksum_',an)));