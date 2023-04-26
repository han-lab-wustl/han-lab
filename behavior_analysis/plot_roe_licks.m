% Zahra
% plot behavior (ROE, licks) of animals in SST experiment
% early days
% assumes that behavior is aligned to Fall.mat
% NOTE: day 1 of experiment should not have any behavior data aligned (no
% VR file, but there is a clampex file)

clear all; close all;
src = 'Z:\sstcre_imaging'; 
animal = 'e201';
fls = dir(fullfile(src, animal, "**\Fall.mat"));
for i=1:length(fls)
    [parent,nm,~] = fileparts(fileparts(fileparts(fileparts(fls(i).folder))));
    dy{i} = str2num(nm);
end
dys = cell2mat(dy);
behavior_days = [15:18, 43:47]; % get behavior for only these days; ZD added for her folder structure
fls = {behavior_days,fls(ismember(dys,behavior_days))}; % only certain days
% 1 is day name, 2 is folder structure
grayColor = [.7 .7 .7];

for fl=1:numel(fls{1})
    flnm = fullfile(fls{2}(fl).folder, fls{2}(fl).name);
    mouse=load(flnm);
    try % don't plot if no behavior data aligned
        tr = mouse.rewards;
    end
    if exist('tr', 'var')==1
        sol = mouse.rewards==0.5; % codes for solenoid
        rew = mouse.rewards>0.5; % codes for single or double rewards
        figure;
        plot(sol*50, 'g', 'LineWidth',3); hold on; 
        plot(rew*50, 'b', 'LineWidth',3); hold on; 
        test = mouse.lickVoltage*1000;
        plot(test,'r'); hold on;
%         plot(find(mouse.licks),test(find(mouse.licks)),'k*')

        plot(mouse.forwardvel, 'Color', grayColor)
        title(sprintf("mouse %s, day %i", animal, fls{1}(fl)))
        xticks(1:1000:numel(mouse.ybinned))
        xticklabels(ceil(mouse.timedFF(1:1000:end))) % plots in seconds
        xlabel("time (s)")
        ylabel("normalized value")
        legend(["solenoid (CS)", "rewards", "lickvoltage", "forward velocity"])
    end
    mice{fl}=mouse;
    clear tr %remove condition from previous loop run
end
%%
% peri CS solenoid 
range=75; % peri triggered FRAMES
recframes = 40000 ;
for d=1:length(fls{1})
    day=mice(d);day=day{1};
    
    rewardsonly=day.rewards==1;
    cs=day.rewards==0.5;
    % runs for all cells
    idx = find(cs);
    periCSvel = zeros(length(idx),(range*2)+1);
    for iid=1:length(idx)
        rn = (idx(iid)-range:idx(iid)+range);
        if max(rn)>recframes% exclude rews towards the end of recording
            rn(find(rn>recframes))=NaN;
        end
        try
            periCSvel(iid,:)=day.licks(rn);
        end
    end
    periCSveld{d}=periCSvel; % per trial
    periCSveld_av{d} = nanmean(periCSvel,1); % average
    [daynm,~] = fileparts(day.ops.data_path);
    [~,daynm] = fileparts(daynm);
    daynms{d}=daynm;
    
end

% plot CS triggered velocity changes
for d=1:length(periCSveld)
    figure;
    try
%         periCSveld{d}(periCSveld{d}==0) = NaN;
%         plot(periCSveld{d}', 'k*'); hold on;          
        plot(periCSveld_av{d}, 'k');
    end
    xlim([0 range*2+1])
    x1=xline(range+1,'-.b',{'Conditioned', 'stimulus'}); %{'Conditioned', 'stimulus'}, 'Reward'
    title(sprintf("%s, day %s", animal , daynms{d}))
    xlabel('frames')
    ylabel('average number of licks')
end
%%
% plot mean image per day
figure;
j = 0; % offset
for i=1:length(mice)
    axes{i-j}=subplot(5,5,i-j);
    imagesc(mice{i}.ops.meanImg) %meanImg or max_proj
    colormap('gray')
    title(sprintf("day %i", i-j))
    axis off;
    hold on;    
    disp(i-j)
    disp(mice{i}.ops.save_path0)
end
linkaxes([axes{:}], 'xy')
