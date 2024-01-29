% Zahra
% analyze behavior of mouse in HRZ memory consolidation experiment
% look at fraction of licks in normal vs. probe trials
% https://www.nature.com/articles/s41593-022-01050-4
close all; clear all;

% select files
sessions = [dir("Y:\hrz_consolidation\behavior\rewloc1\*.mat"); dir("Y:\hrz_consolidation\behavior\rewloc3\*.mat")];
prevrewloc = [ones(1,length(dir("Y:\hrz_consolidation\behavior\rewloc1\*.mat")))*101 ones(1,length(dir("Y:\hrz_consolidation\behavior\rewloc3\*.mat")))*151]; 
newrewloc = [ones(1,length(dir("Y:\hrz_consolidation\behavior\rewloc1\*.mat")))*101 ones(1,length(dir("Y:\hrz_consolidation\behavior\rewloc3\*.mat")))*151]; 
ind = 1;
% days = filename;
grayColor = [.7 .7 .7]; coms_init = {}; coms_btwn = {}; coms_learn = {};

for dy=1:length(sessions)
    mouse_pth = sessions(dy);
    mouse = load(fullfile(mouse_pth.folder, mouse_pth.name));    
    mouse.VR.lick(mouse.VR.ypos<3)=0; % filter dark time licks
    % get success and fail trials
    [s,f,str, ftr, ttr, tr] = get_success_failure_trials(mouse.VR.trialNum, mouse.VR.reward);
    success_prop{ind} = s/tr;
    fail_prop{ind} = f/tr;    
    eps = find(mouse.VR.changeRewLoc>0);
    eps = [eps length(mouse.VR.changeRewLoc)]; 
    rewlocs = unique(mouse.VR.changeRewLoc);
    gainf = 1/mouse.VR.scalingFACTOR;
    rewloc = rewlocs(rewlocs>0)*gainf;
    rewsize = mouse.VR.settings.rewardZone*gainf;
    fig = figure('Renderer', 'painters');
    ypos = mouse.VR.ypos*(gainf);
    velocity = mouse.VR.ROE(2:end)*-0.013./diff(mouse.VR.time);
    plot(ypos, 'Color', grayColor, 'LineWidth',1.2); hold on; 
    plot(find(mouse.VR.lick),ypos(find(mouse.VR.lick)), ...
        'r.', 'MarkerSize',10) 
    for mm = 1:length(eps)-1 %the rectangle indicating the reward location, overlaps the probe trials referring to the previous reward location
        rectangle('position',[eps(mm) rewloc-rewsize/2 ...
            eps(mm+1)-eps(mm) rewsize],'EdgeColor',[0 0 0 0],'FaceColor',[.7 .7 .7 .4])
    end
    plot(find(mouse.VR.reward==0.5),ypos(find(mouse.VR.reward==0.5)),'k*', ...
        'MarkerSize',15)
    ylabel("Track Position (cm)")    
    xticks([0:10000:length(mouse.VR.time)])
    tic = floor(mouse.VR.time(1:10000:end)/60);
    xticklabels(tic)
    xlabel("Time (minutes)")
    ylim([0 270])
    xlim([0 length(mouse.VR.reward)])
    yticks([0:90:270])
    yticklabels([0:90:270])
    sgtitle(string(mouse_pth.name))
%     legend({'Position', 'Licks', 'Conditioned Stimulus'})
%     saveas(fig, 'C:\Users\Han\Box\neuro_phd_stuff\han_2023\dlc\dlc_poster_2023\behavior.svg')
%     close(fig)    
    success = 0; % probes
    rewsize = 20;
    % plot com in initial probes vs. b/wn epoch probes    
    eps = find(mouse.VR.changeRewLoc>0);
    eps = [eps length(mouse.VR.changeRewLoc)];
    trialnum = mouse.VR.trialNum(eps(1):eps(2)); reward = mouse.VR.reward(eps(1):eps(2));    
    licks = mouse.VR.lick(eps(1):eps(2)); ypos = mouse.VR.ypos(eps(1):eps(2));
    [success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum,reward);
    % get regular coms?
    trials = ttr; % all trials (non probes)
    trials_ = []; trind = 1;
    for tr=trials
        if sum(licks(trialnum==tr))>0
            trials_(trind)=tr;
        end
        trind=trind+1;
    end
    [com] = get_com_licks(trialnum, reward, trials, logical(licks), ypos, newrewloc(dy), ...
        rewsize, success);
    coms_learn{ind} = com;
    trials = unique(trialnum(trialnum<min(str))); % get trials before first successful trial
    % get trials only when the mouse licks
    trials_ = []; trind = 1;
    for tr=trials
        if sum(licks(trialnum==tr))>0
            trials_(trind)=tr;
        end
        trind=trind+1;
    end
    [com] = get_com_licks(trialnum, reward, trials, logical(licks), ypos, prevrewloc(dy), ...
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
        [com] = get_com_licks(trialnum, reward, trials, logical(licks), ypos, newrewloc(dy), ...
            rewsize, success);
        coms_btwn{ind} = mean(com, 'omitnan');
    end
    ind=ind+1;    
end
%%
% collect coms from hrz probes
load('C:\Users\Han\Box\neuro_phd_stuff\han_2023-\ed_grant_2024\com_hrz_probes.mat')
lastep_mask = cell2mat(cellfun(@length, COMlick_rewlocprev, 'UniformOutput',false));
lastep_COM = {};
for ii=1:length(COMlick_rewlocprev)
    ddd = cell2mat(COMlick_rewlocprev(ii));
    disp(ddd)
    lastep_COM{ii} = ddd(lastep_mask(ii));
end
%%
nulldist = cell2mat(lastep_COM(~cellfun('isempty',lastep_COM)));
randomsess = unique(randi([1 length(nulldist)],1,17));
nulldist = nulldist(randomsess);
% plot com in initial probes vs. b/wn epoch probes    
figure;
bar([mean(nulldist, 'omitnan') NaN mean(cell2mat(cellfun(@(x) mean(x, 'omitnan'), coms_init, 'UniformOutput', false))) NaN... % mean of all trials
    mean(cell2mat(cellfun(@(x) mean(x, 'omitnan'), coms_learn, 'UniformOutput', false))) NaN...% mean of all trials
    mean(cell2mat(coms_btwn))], 'FaceColor', 'w'); hold on
x = [ones(1,length(nulldist)), ones(1,length(cell2mat(cellfun(@(x) mean(x, 'omitnan'), coms_init, 'UniformOutput', false))))*3,...
    ones(1,length(cell2mat(cellfun(@(x) mean(x, 'omitnan'), coms_learn, 'UniformOutput', false))))*5,...
    ones(1,length(cell2mat(coms_btwn)))*7];
y = [nulldist, cell2mat(cellfun(@(x) mean(x, 'omitnan'), coms_init, 'UniformOutput', false)),...
    cell2mat(cellfun(@(x) mean(x, 'omitnan'), coms_learn, 'UniformOutput', false)),...
    cell2mat(coms_btwn)];
swarmchart(x,y,'ko')
ylabel('Center of Mass Licks (cm)-Reward Location')
xlabel('Trial Type')
xticklabels(["Non-memory Probes", "", "Initial Probes", "", "Learning Trials", "", "Between Epoch Probes"])

initprobes = cell2mat(cellfun(@(x) mean(x, 'omitnan'), coms_init, 'UniformOutput', false));
[h,p,i,stats] = ttest2(abs(initprobes), abs(nulldist));
title(sprintf('memory task: n=2 animals, 9 sessions \n p=%f bwn hrz probes and memory probes', p))
box off