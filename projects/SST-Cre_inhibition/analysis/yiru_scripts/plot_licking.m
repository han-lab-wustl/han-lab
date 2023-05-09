% need to change ylim to 0-270, rewrange, ybinned-involved calculation for
% actual world.

clearvars
Settings.paths = dir('Y:\E186\E186\D*\Fall.mat');
Settings.level_mouse_name = 3;
Settings.level_day = 4;
dates = strings(1, size(Settings.paths,1));

%% Create faliure plot for each day
for this_day = 1:size(Settings.paths,1)

    file = fullfile(Settings.paths(this_day).folder,Settings.paths(this_day).name);
    directory = file;
    info = split(directory,'\');
    mouse_cd = string(info{Settings.level_mouse_name});
    day_cd = string(info{Settings.level_day});

    l = load(file);
    vr = l.VR;
    date = extractAfter(extractBefore(vr.name_date_vr, "_time"), "E186_");
    specific_date = char(datetime(date,'InputFormat','dd_MMM_yyyy','Format','yyyy-MM-dd'));
    dates(this_day) = specific_date + " (" + day_cd + ")";
    
    rewrange = 10;% *2/3; %rewardzonesize = 10; set 10 to make sure it covers all rewards
    rewloc = find(l.changeRewLoc ~= 0);
    rewypos = l.changeRewLoc(rewloc);
%     all_epocvec(this_day,1:length(rewypos)) = 1:length(rewypos);
%     all_rewypos(this_day, 1:length(rewypos)) = rewypos;
    epochs = "ep" + string(1: length(rewloc));
    loc = [rewloc, size(l.changeRewLoc, 2)];
    f = figure('Renderer','painters','Position', [20 20 2000 1500], 'visible', 'off');
    
    for idx = 1:length(rewloc)
        rewl = l.changeRewLoc(1, loc(idx))*1.5;
        single_epoch_frames = loc(idx):loc(idx+1)-1;
        probes_index = find(l.trialnum < 3);
        probes_frames = probes_index(ismember(probes_index, single_epoch_frames));        
        lick_index = find(l.licks);
        reward_index = find(l.rewards);
        lick_idx = lick_index(ismember(lick_index,single_epoch_frames)); %same as (lick_index>loc(idx) & lick_index<=loc(idx+1))
        reward_idx = reward_index(ismember(reward_index, single_epoch_frames));
        time_ep = l.timedFF(single_epoch_frames); %(non_probes_frames(2:end));
        ypos_ep = l.ybinned(single_epoch_frames)*1.5; % gain
        subplot(length(rewloc),1,idx);

        % time as x_axis
        if ~isempty(time_ep)
            patch([time_ep(1) time_ep(end) time_ep(end) time_ep(1)],[rewl-rewrange/2 rewl-rewrange/2 rewl+rewrange/2 rewl+rewrange/2],[0 0 0.4],'FaceAlpha',.2);
            hold on;
        end
        plot(time_ep,ypos_ep,'Color', [0.5 0.5 0.5]);%%
        ylim([0 270]);
        hold on;
        if ~isempty(probes_frames)
            patch([l.timedFF(probes_frames(1)) l.timedFF(probes_frames(end)) l.timedFF(probes_frames(end)) l.timedFF(probes_frames(1))],[0 0 270 270],'b','FaceAlpha',0.03,'EdgeColor','b');
        else
            patch([time_ep(1) time_ep(1) time_ep(1) time_ep(1)],[0 0 0 0],'b','FaceAlpha',0.03,'EdgeColor','b');
        end
        hold on;
        scatter(l.timedFF(lick_idx),l.ybinned(lick_idx)*1.5, 3, 'r');
        scatter(l.timedFF(reward_idx),l.ybinned(reward_idx)*1.5, 10, 'k','filled');
        hold on;
        if isfield(vr, 'optotrigger')
            opto_index = find(vr.optotrigger == 1);
            opto_index = opto_index(opto_index >= loc(idx) & opto_index <= loc(idx+1));
            if ~isempty(opto_index)
                patch([l.timedFF(opto_index(1)) l.timedFF(opto_index(end)) l.timedFF(opto_index(end)) l.timedFF(opto_index(1))],[0 0 270 270],'g','FaceAlpha',0.08,'EdgeColor','g');
            else
                patch([time_ep(1) time_ep(1) time_ep(1) time_ep(1)],[0 0 0 0],'g','FaceAlpha',0.08,'EdgeColor','g');
            end
            legend('Reward Zone','Trace','Probe trials','Licking','Reward','Opto Stim', 'Location', 'eastoutside');
        else
            legend('Reward Zone','Trace','Probe trials','Licking','Reward','Location', 'eastoutside');
        end

        title("Reward y = "+rewl+' cm')
        ylabel({['ep' num2str(idx)], 'y position (cm)'},'FontSize',15);
        %legend('Reward Zone','Trace','Probe trials','Licking','Reward','Opto Stim', 'Location', 'eastoutside')
        hold off;
    end
    xlabel({' ', 'time(s)'}, 'FontSize', 18);
    sgtitle({'Licking along the track on '+day_cd+" ("+specific_date+"): "+mouse_cd,' '}, 'FontSize', 18);
    if isfield(vr, 'optotrigger')
        day_cd = string([char(day_cd) '(Opto)']);
    end

%% Save the figure
%     set(f, 'PaperUnits', 'inches');
%     x_width=24;y_width=12;
%     set(f, 'PaperPosition', [0 0 x_width y_width]); 
    saveas(f,['H:\E186\E186\licking_fig\png\' char(day_cd) '_licking.png'],'png');
    saveas(f,['H:\E186\E186\licking_fig\mat\' char(day_cd) '_licking.m'],'m');
    saveas(f,['H:\E186\E186\licking_fig\svg\' char(day_cd) '_licking.svg'],'svg');
%     hold off;
    disp([file ' Done!']);
end
disp('All done!');