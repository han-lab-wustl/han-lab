%% Create arrays for Dxx Date Reward-location
Settings.paths = dir('Y:\E186\E186\D*\Fall.mat');
Settings.level_mouse_name = 3;
Settings.level_day = 4;
%dates = strings(1, size(Settings.paths,1));
%days = strings(1,size(Settings.paths,1));
rewardloc = strings(4*size(Settings.paths,1),4); %date Dxx ep rewypos
for this_d = 1:size(Settings.paths,1)
    file = fullfile(Settings.paths(this_d).folder,Settings.paths(this_d).name);
    directory = file;
    info = split(directory,'\');
    mouse_cd = string(info{Settings.level_mouse_name});
    day_cd = string(info{Settings.level_day});
    l = load(file);
    vr = l.VR;
    date = extractAfter(extractBefore(vr.name_date_vr, "_time"), "E186_");
    specific_date = char(datetime(date,'InputFormat','dd_MMM_yyyy','Format','yyyy-MM-dd'));
    %rewrange = 10 *2/3; %rewardzonesize = 10; set 10 to make sure it covers all rewards
    rewloc = find(l.changeRewLoc ~= 0);
    rewypos = l.changeRewLoc(rewloc);
    for i = 2:length(rewloc)-1
        rewardloc(4*this_d+i-5,1) = specific_date;
        rewardloc(4*this_d+i-5,2) = char(day_cd);
        rewardloc(4*this_d+i-5,3) = string(i);
        rewl = rewypos(i); %l.changeRewLoc(1, rewloc(i))
        rewardloc(4*this_d+i-5,4) = rewl;
    end
end
rewardloc(strcmp(rewardloc(:,4), ''), :) = [];
[~,idx] = sort(str2double(rewardloc(:,4)));
dates = rewardloc(:,1);
days = rewardloc(:,2);
eps = rewardloc(:,3);
rewards = str2double(rewardloc(:,4));
rewardloc(:,1) = dates(idx);
rewardloc(:,2) = days(idx);
rewardloc(:,3) = eps(idx);
rewardloc(:,4) = rewards(idx);
disp('Sorting done!')
%% loop over reward locations
for n = 1:ceil(size(rewardloc,1)/5)
    f = figure('visible','off','Renderer','painters','Position', [20 20 2000 1500]);
    if n~=ceil(size(rewardloc,1)/5)
        these_days = rewardloc((5*n-4):5*n,:);
    else
        these_days = rewardloc(5*n-4:length(rewardloc),:);
    end
    dates_n = these_days(:,1);
    days_n = these_days(:,2);
    eps_n = these_days(:,3);
    rewards_n = these_days(:,4);
    for this_day = 1:size(these_days,1)
        file = 'Y:\E186\E186\'+days_n(this_day)+'\Fall.mat';
        %Settings.paths = dir([filename]);
        %Settings.level_mouse_name = 3;
        %Settings.level_day = 4;
        %dates = strings(1, size(Settings.paths,1));
        %for this_day = 1:size(Settings.paths,1)
            %file = fullfile(Settings.paths(this_day).folder,Settings.paths(this_day).name);
        directory = file;
        info = split(directory,'\');
        mouse_cd = string(info{Settings.level_mouse_name});
        day_cd = string(info{Settings.level_day});
        
        l = load(file);
        vr = l.VR;
        %dates(this_day) = specific_date + " (" + day_cd + ")";
        %date = specific_date + " (" + day_cd + ")";
            
            %days(this_day) = char(day_cd)
            
        rewrange = 10; %*2/3; %rewardzonesize = 10; set 10 to make sure it covers all rewards
        rewloc = find(l.changeRewLoc ~= 0);
        %rewypos = l.changeRewLoc(rewloc);
        rewl = str2double(rewards_n(this_day))*1.5;
        %     all_epocvec(this_day,1:length(rewypos)) = 1:length(rewypos);
        %     all_rewypos(this_day, 1:length(rewypos)) = rewypos;
        epoch = "ep" + eps_n(this_day);
            %loc = [rewloc, size(l.changeRewLoc, 2)];
            %f = figure('visible', 'off');
        startframe = rewloc(str2double(eps_n(this_day)));
        endframe = rewloc(str2double(eps_n(this_day))+1)-1;
        single_epoch_frames = startframe:endframe;
        probes_index = find(l.trialnum < 3);
        probes_frames = probes_index(ismember(probes_index, single_epoch_frames));        
        lick_index = find(l.licks);
        reward_index = find(l.rewards);
        lick_idx = lick_index(ismember(lick_index,single_epoch_frames)); %same as (lick_index>loc(idx) & lick_index<=loc(idx+1))
        reward_idx = reward_index(ismember(reward_index, single_epoch_frames));
        time_ep = l.timedFF(single_epoch_frames); %(non_probes_frames(2:end));
        ypos_ep = l.ybinned(single_epoch_frames)*1.5; % gain
        subplot(length(these_days),1,this_day);
        patch([time_ep(1) time_ep(end) time_ep(end) time_ep(1)],[rewl-rewrange/2 rewl-rewrange/2 rewl+rewrange/2 rewl+rewrange/2],[0 0 0.4],'FaceAlpha',.2);
        hold on    
        plot(time_ep,ypos_ep,'Color', [0.5 0.5 0.5]);
        ylim([0 270]); %ylim([0 180]);
        hold on
        patch([l.timedFF(probes_frames(1)) l.timedFF(probes_frames(end)) l.timedFF(probes_frames(end)) l.timedFF(probes_frames(1))],[0 0 270 270],'b','FaceAlpha',0.03,'EdgeColor','b');
        hold on
        scatter(l.timedFF(lick_idx),l.ybinned(lick_idx)*1.5, 3, 'r');
        scatter(l.timedFF(reward_idx),l.ybinned(reward_idx)*1.5, 10, 'k','filled');
        hold on
        if isfield(vr, 'optotrigger')
            opto_index = find(vr.optotrigger == 1);
            opto_index = opto_index(opto_index >= startframe & opto_index <= endframe);
            day_cd = string([char(day_cd) '(Opto)']);
            if ~isempty(opto_index)
                patch([l.timedFF(opto_index(1)) l.timedFF(opto_index(end)) l.timedFF(opto_index(end)) l.timedFF(opto_index(1))],[0 0 270 270],'g','FaceAlpha',0.08,'EdgeColor','g');
            else
                patch([time_ep(1) time_ep(1) time_ep(1) time_ep(1)],[0 0 0 0],'g','FaceAlpha',0.08,'EdgeColor','g');
            end
            legend('Reward Zone','Trace','Probe trials','Licking','Reward','Opto Stim', 'Location', 'eastoutside');
        else
            legend('Reward Zone','Trace','Probe trials','Licking','Reward','Location', 'eastoutside');
        end
        xlabel('Time (s)');
        ylabel('y position (cm)');
        title('Licking of '+epoch+' on '+day_cd+" ("+dates_n(this_day)+"), Reward y = "+rewl+' cm');
        hold on
    end
    hold off
    
    % Save the figure
%     set(f, 'PaperUnits', 'inches');
%     x_width=24;y_width=18;
%     set(f, 'PaperPosition', [0 0 x_width y_width]); 
    saveas(f,['H:\E186\E186\licking_fig\png\licking_comparison' num2str(n) '.png'],'png');
    saveas(f,['H:\E186\E186\licking_fig\mat\licking_comparison' num2str(n) '.m'],'m');
    saveas(f,['H:\E186\E186\licking_fig\svg\licking_comparison' num2str(n) '.svg'],'svg');
end
disp('All done!');