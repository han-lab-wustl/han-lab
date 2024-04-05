% lick performance opto
% take abs value of distance of lick from start of rewloc --> no diff in
% coms
% look at ratio of licks pre-reward - top 25% itle
% av across the entire epoch

clear all; close all
% mouse_name = "e216";
mice = ["e216", "e218", "e217", "e189", "e190", "e201", "e186"];
cond = ["vip", "vip", "ctrl", "ctrl", "sst", "sst"];%, "pv"];
dys_s = {[7 8 9 37 38 39 40 41 42 44 45 46 48 50:59], ...
    [20,21,22,23,35,36,37,38,39,40,41,...
     42,43,44,45,47 48 49 50 51 52 55 56], [2:11],...
     [35:42,44],...
     [33:35, 40:43, 45]...
     [52:59], [2:5,31,32,33]};
% experiment conditions: preopto=-1; optoep=3/2; control day1=0; control
% day2=1
opto_eps = {[-1 -1 -1 2 -1 0 1 3 -1 -1 0 1 2 3 0 1 2 3 0 1 2 0 1],...
    [-1 -1 -1 -1,3 0 1 2 0 1 3,0 1 2, 0 3 0 1 2 0 1 2 0], [-1 -1 -1 -1 2 3 2 0 3 0], ...
    [-1 -1 -1 -1 2 3 2 0 2],...
    [-1 -1 -1 3 0 1 2 3],...
    [-1 -1 -1 2 3 0 2 3],...
    [-1 -1 -1 -1 2 3 2]};
src = ["X:\vipcre", "X:\vipcre", "X:\vipcre", '\\storage1.ris.wustl.edu\ebhan\Active\dzahra', ...
    '\\storage1.ris.wustl.edu\ebhan\Active\dzahra', 'Z:\sstcre_imaging', ...
    'Y:\analysis\fmats'];


licks_m = {};
for m=1:length(mice)
dys = dys_s{m};
opto_ep = opto_eps(m); opto_ep = opto_ep{1};
epind = 1; % for indexing
% ntrials = 8; % get licks for last n trials
licks_opto = []; licks_ctrl = []; licks_preopto = []; licks_inctrl_1 = []; licks_inctrl_2 = []; licks_postopto = [];
for dy=dys
%     daypth = dir(fullfile(src, mouse_name, string(dy), "**\*Fall.mat"));
%     load(fullfile(daypth.folder,daypth.name), 'licks', 'trialnum', 'rewards', 'changeRewLoc', ...
%         'ybinned', 'timedFF', 'VR');
    daypth = dir(fullfile(src(m), mice(m), string(dy), '**', '*time*.mat'));    
    if m==7 % e186 in a diff format
        daypth = dir(fullfile(src(m), mice(m), 'days', '*.mat'));
        daypth = daypth(dy);    
        load(fullfile(daypth.folder, daypth.name), 'VR') % load fall
    else
        load(fullfile(daypth.folder, daypth.name))
    end
    eps = find(VR.changeRewLoc>0);
    eps = [eps length(VR.changeRewLoc)];
    track_length = 180/VR.scalingFACTOR;
    ybinned = VR.ypos/VR.scalingFACTOR;
    rewlocs = VR.changeRewLoc(VR.changeRewLoc>0)/VR.scalingFACTOR;
    rewsize = VR.settings.rewardZone/VR.scalingFACTOR;
    trialnum = VR.trialNum; % use VR variables
    rewards = VR.reward;
    time = VR.time;
    licks = logical(VR.lick);
    if opto_ep(epind)==3
        eprng = eps(3):eps(4);
        [lickbin_s,lickbin_f,prerewlickbin,prerewlickbin_ratio_opto] = get_pre_reward_lick_binned(track_length, ...
    eprng, trialnum, rewards, licks, ybinned, opto_ep(epind), rewlocs, time );
        % vs. previous epoch
        eprng = eps(2):eps(3);
        [lickbin_s,lickbin_f,prerewlickbin,prerewlickbin_ratio_preopto] = get_pre_reward_lick_binned(track_length, ...
    eprng, trialnum, rewards, licks, ybinned, 2, rewlocs, time);
        % vs. next ep
        if length(eps)>4
            eprng = eps(4):eps(5);
            [lickbin_s,lickbin_f,prerewlickbin,prerewlickbin_ratio_postopto] = get_pre_reward_lick_binned(track_length, ...
        eprng, trialnum, rewards, licks, ybinned, 4, rewlocs, time);    
        else
            prerewlickbin_ratio_postopto = 0;
        end

        licks_opto(epind) = prerewlickbin_ratio_opto;
        licks_postopto(epind) = prerewlickbin_ratio_postopto;
        licks_ctrl(epind) = prerewlickbin_ratio_preopto;
    elseif opto_ep(epind)==2
        eprng = eps(2):eps(3);
        [lickbin_s,lickbin_f,prerewlickbin,prerewlickbin_ratio_opto] = get_pre_reward_lick_binned(track_length, ...
    eprng, trialnum, rewards, licks, ybinned, opto_ep(epind), rewlocs, time);
        % vs. previous epoch
        eprng = eps(1):eps(2);
        [lickbin_s,lickbin_f,prerewlickbin,prerewlickbin_ratio_preopto] = get_pre_reward_lick_binned(track_length, ...
    eprng, trialnum, rewards, licks, ybinned, 1, rewlocs, time);
        % vs. next ep
        if length(eps)>3
            eprng = eps(3):eps(4);
            [lickbin_s,lickbin_f,prerewlickbin,prerewlickbin_ratio_postopto] = get_pre_reward_lick_binned(track_length, ...
        eprng, trialnum, rewards, licks, ybinned, 3, rewlocs, time);    
        else
            prerewlickbin_ratio_postopto = 0;
        end
        licks_opto(epind) = prerewlickbin_ratio_opto;
        licks_postopto(epind) = prerewlickbin_ratio_postopto;
        licks_ctrl(epind) = prerewlickbin_ratio_preopto;

    elseif opto_ep(epind)==-1 % just pre opto days
        eprng = eps(2):eps(3);
        [lickbin_s,lickbin_f,prerewlickbin,prerewlickbin_ratio] = get_pre_reward_lick_binned(track_length, ...
    eprng, trialnum, rewards, licks, ybinned, 2, rewlocs, time);
        licks_preopto(epind) = prerewlickbin_ratio;
    elseif opto_ep(epind)==0  % intermediate control days 1
        eprng = eps(2):eps(3);
        [lickbin_s,lickbin_f,prerewlickbin,prerewlickbin_ratio] = get_pre_reward_lick_binned(track_length, ...
    eprng, trialnum, rewards, licks, ybinned, 2, rewlocs, time);
        licks_inctrl_1(epind) = prerewlickbin_ratio;
    elseif opto_ep(epind)==1  % intermediate control days 2
        eprng = eps(2):eps(3);
        [lickbin_s,lickbin_f,prerewlickbin,prerewlickbin_ratio] = get_pre_reward_lick_binned(track_length, ...
    eprng, trialnum, rewards, licks, ybinned, 2, rewlocs, time);
        licks_inctrl_2(epind) = prerewlickbin_ratio;
    end
    epind = epind+1;   
end

%remove zeros from other days
licks_preopto = nonzeros(licks_preopto);
licks_postopto = nonzeros(licks_postopto);
licks_opto = nonzeros(licks_opto);
licks_ctrl = nonzeros(licks_ctrl);
licks_inctrl_1 = nonzeros(licks_inctrl_1);
licks_inctrl_2 = nonzeros(licks_inctrl_2);
licks_m{m} = {{licks_ctrl},{licks_opto},{licks_preopto}, {licks_postopto},{licks_inctrl_1},{licks_inctrl_2}};
end
%%
figure; 
ctrl=cellfun(@(x) x{1}, licks_m); % grab for each mouse
opto=cellfun(@(x) x{2}, licks_m);
offopsin = [mean(ctrl{1}', 'omitnan') mean(ctrl{2}', 'omitnan') mean(ctrl{3}', 'omitnan')];
offvector = [mean(ctrl{4}', 'omitnan') mean(ctrl{5}', 'omitnan') mean(ctrl{6}', 'omitnan') mean(ctrl{7}', 'omitnan')];
onopsin = [mean(opto{1}', 'omitnan') mean(opto{2}', 'omitnan') mean(opto{3}', 'omitnan')];
onvector = [mean(opto{4}', 'omitnan') mean(opto{5}', 'omitnan') mean(opto{6}', 'omitnan') mean(opto{7}', 'omitnan')];
x = [ones(1,length(offvector)), ...
    ones(1,length(onvector))*3, ones(1,length(offopsin))*5, ones(1,length(onopsin))*7];
y = [offvector, onvector, offopsin, onopsin];
means = [mean(offvector,'omitnan') NaN...
    mean(onvector,'omitnan') NaN...
    mean(offopsin,'omitnan') NaN...
    mean(onopsin,'omitnan')];
bar(means, 'FaceColor', 'w'); hold on
swarmchart(x,y,'k'); hold on
yerr = {offopsin,NaN,offvector,NaN,onopsin,NaN,onvector};
err = [];
for i=1:length(yerr)
    err(i) =(std(yerr{i},'omitnan')/sqrt(size(yerr{i},2))); 
end
er = errorbar([1 NaN 3 NaN 5 NaN 7],means,err);
er.Color = [0 0 0];                            
er.LineStyle = 'none';  
[h,p,i,stats]=ttest(offopsin,onopsin)
ylabel("Lick Rate / Trial")
xticklabels(["Control LED off", "", "Control LED on", "", "VIP stGtACR LED off", ...
    "", "VIP stGtACR LED on"])
condt = [repelem("Control LED off",length(offopsin)), repelem("Control LED on",length(onopsin)), ...
    repelem("VIP stGtACR LED off",length(offvector)), ...
    repelem("VIP stGtACR LED on",length(onvector))]';
tbl = table(condt,[offvector,onvector,offopsin,onopsin]',VariableNames=["Condition" "Lick Rate"]);
aov = anova(tbl,'Lick Rate');
multcompare(aov,'CriticalValueType',"bonferroni")
box off
diffpow = mean(onopsin);
basepwr = mean(onvector);
stdpow = std([onvector onopsin]);
nout = sampsizepwr('t2',[basepwr, stdpow],diffpow,0.80);
% pwrout = sampsizepwr('t2',[basepwr, stdpow],diffpow,[],7);
halfeffect = (basepwr-diffpow)/2;
nout_half = sampsizepwr('t2',[basepwr, stdpow],basepwr-halfeffect,0.80);