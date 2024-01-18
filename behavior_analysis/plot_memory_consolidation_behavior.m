% Zahra
% analyze behavior of mouse in HRZ memory consolidation experiment
% look at fraction of licks in normal vs. probe trials
% https://www.nature.com/articles/s41593-022-01050-4
close all; clear all;

% select files
[filename,filepath] = uigetfile('*.mat','MultiSelect','on');

ind = 1;
days = filename;
grayColor = [.7 .7 .7]; coms_init = {}; coms_btwn = {};
for dy=1:length(days)
    mouse_pth = fullfile(filepath,filename{dy});
    mouse = load(mouse_pth);    
    % get success and fail trials
    [s,f,str, ftr, ttr, tr] = get_success_failure_trials(mouse.VR.trialNum, mouse.VR.reward);
    success_prop{ind} = s/tr;
    fail_prop{ind} = f/tr;    
    eps = find(mouse.VR.changeRewLoc>0);
    eps = [eps length(mouse.VR.changeRewLoc)]; 
    gainf = 1/mouse.VR.scalingFACTOR;
    rewloc = mouse.VR.changeRewLoc(mouse.VR.changeRewLoc>0)*gainf;
    rewsize = mouse.VR.settings.rewardZone*gainf;
    fig = figure('Renderer', 'painters');
    ypos = mouse.VR.ypos*(gainf);
    velocity = mouse.VR.ROE(2:end)*-0.013./diff(mouse.VR.time);
    scatter(1:length(ypos), ypos, 2, 'filled', 'MarkerFaceColor', grayColor); hold on; 
    plot(find(mouse.VR.lick),ypos(find(mouse.VR.lick)), ...
        'r.', 'MarkerSize',5) 
    for mm = 1:length(eps)-1 %the rectangle indicating the reward location, overlaps the probe trials referring to the previous reward location
        rectangle('position',[eps(mm) rewloc(mm)-rewsize/2 ...
            eps(mm+1)-eps(mm) rewsize],'EdgeColor',[0 0 0 0],'FaceColor',[0 .5 .5 0.3])
    end
    plot(find(mouse.VR.reward==0.5),ypos(find(mouse.VR.reward==0.5)),'b*', ...
        'MarkerSize',5)
    ylabel("Track Position (cm)")    
    xticks([0:10000:length(mouse.VR.time)])
    tic = floor(mouse.VR.time(1:10000:end)/60);
    xticklabels(tic)
    xlabel("Time (minutes)")
    ylim([0 270])
    xlim([0 length(mouse.VR.reward)])
    yticks([0:90:270])
    yticklabels([0:90:270])
    sgtitle(string(filename{dy}))
%     legend({'Position', 'Licks', 'Conditioned Stimulus'})
%     saveas(fig, 'C:\Users\Han\Box\neuro_phd_stuff\han_2023\dlc\dlc_poster_2023\behavior.svg')
%     close(fig)    
    rewloc = 101; % fixed
    success = 0; % probes
    rewsize = 20;
    % plot com in initial probes vs. b/wn epoch probes    
    eps = find(mouse.VR.changeRewLoc>0);
    eps = [eps length(mouse.VR.changeRewLoc)];
    trialnum = mouse.VR.trialNum(eps(1):eps(2)); reward = mouse.VR.reward(eps(1):eps(2));
    licks = mouse.VR.lick(eps(1):eps(2)); ypos = mouse.VR.ypos(eps(1):eps(2));
    [success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum,reward);
    trials = unique(trialnum(trialnum<min(str))); % get trials before first successful trial
    % get trials only when the mouse licks
    trials_ = []; trind = 1;
    for tr=trials
        if sum(licks(trialnum==tr))>0
            trials_(trind)=tr;
        end
        trind=trind+1;
    end
    [com] = get_com_licks(trialnum, reward, trials, logical(licks), ypos, rewloc, ...
        rewsize, success);
    coms_init{ind} = com;
    % get ep1 probe coms
    if length(eps)>2
        trialnum = mouse.VR.trialNum(eps(2):eps(3)); reward = mouse.VR.reward(eps(2):eps(3));
        licks = mouse.VR.lick(eps(2):eps(3)); ypos = mouse.VR.ypos(eps(2):eps(3));
        [success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum,reward);
        trials = unique(trialnum(trialnum<3)); % get trials before first successful trial
        % get trials only when the mouse licks
        trials_ = []; trind = 1;
        for tr=trials
            if sum(licks(trialnum==tr))>0
                trials_(trind)=tr;
            end
            trind=trind+1;
        end
        [com] = get_com_licks(trialnum, reward, trials, logical(licks), ypos, rewloc, ...
            rewsize, success);
        coms_btwn{ind} = mean(com, 'omitnan');
    end
    ind=ind+1;    
end

% plot com in initial probes vs. b/wn epoch probes    
figure;
bar([mean(cell2mat(cellfun(@(x) mean(x, 'omitnan'), coms_init, 'UniformOutput', false))) mean(cell2mat(coms_btwn))], 'FaceColor', 'w'); hold on
plot(1, cell2mat(cellfun(@(x) mean(x, 'omitnan'), coms_init, 'UniformOutput', false)), 'ko')
plot(2, cell2mat(coms_btwn), 'ko')
ylabel('COM licks')
xlabel('probes')
xticklabels(["initial probes", "b/wn epoch probes"])
