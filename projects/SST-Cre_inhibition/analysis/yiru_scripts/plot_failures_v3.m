% This version is used to look at opto days and control days.

%% create faliure plot for each day
Settings.paths = dir('Y:\E186\E186\D*\Fall.mat');% Or Change drive name if switch the PC
Settings.level_mouse_name = 3;
Settings.level_day = 4;
%total_failure_success_days = int16.empty; %%%
dates = strings(1, size(Settings.paths,1));
all_rewypos = zeros(size(Settings.paths,1),4);
all_epocvec = zeros(size(Settings.paths,1),4);
summary_days = zeros(length(Settings.paths), 6); %fail_opto# fail_nonopto# success_opto# success_nonopto# opto# total#
failure_percent_days = zeros(length(Settings.paths),7); %opto_f/opto nonopto_f/nonopto_f opto_f/total nonopto_f/total opto/total nonopto/total f/total

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
    
    rewrange = 10 *2/3; %rewardzonesize = 10; set 10 to make sure it covers all rewards
    rewloc = find(l.changeRewLoc ~= 0);
    summary_eps = zeros(length(rewloc), 7); % ep# fail_opto# fail_nonopto# success_opto# success_nonopto# opto# total#
    summary_eps(:,1) = 1:length(rewloc);
    total_failure_success_epochs = zeros(length(rewloc), 3);
    loc = [rewloc, size(l.changeRewLoc, 2)];
    for idx = 1:length(rewloc)
        rewl = l.changeRewLoc(1, loc(idx));
        rl = find(l.ybinned > rewl-rewrange & l.ybinned < rewl+rewrange & l.rewards == 1);
        rl = rl(rl < loc(idx+1) & rl >= loc(idx));
        success_trials = unique(l.trialnum(rl));
        success_per_epoch = length(success_trials);
        single_epoch = l.trialnum(loc(idx):loc(idx+1)-1);
        valid_trials = [2 single_epoch(single_epoch > 2)];
        trialn= valid_trials(end)-2;
        trialmax = valid_trials(end);
        %rewtrialmax= max(l.trialnum(rl));
        last_trial = find(l.trialnum == trialmax & l.ybinned ~= 1);
        last_trial = last_trial(ismember(last_trial, loc(idx)+1:loc(idx+1)));
        if length(unique(l.ybinned(last_trial))) == 1 % && ~isempty(last_trial)
            trialn = trialn-1;
        elseif max(l.ybinned(last_trial))<rewl-rewrange/2%~isempty(last_trial) && sum(l.ybinned(last_trial)>=rewl-rewrange/2)<1 % remove the last one without reaching the reward zone
                trialn = trialn-1;
        end
        
         %%% Check opto stimulation
        if isfield(vr, 'optotrigger')
            opto_index = find(vr.optotrigger == 1);
            opto_index = opto_index(opto_index >= loc(idx) & opto_index <= loc(idx+1));
            if ~isempty(opto_index)
                opto_trials = l.trialnum(opto_index);
                uopto_trials = unique(opto_trials) ;
                %filter out the extra trial
                %hard code
                if length(uopto_trials) ~= 2 && length(uopto_trials) ~= 5
                    uopto_trials = uopto_trials(~isoutlier(histc(opto_trials, unique(opto_trials)),"percentiles",[20 100]));
                end
                uopto_ep_trials = uopto_trials(uopto_trials>2);
                success_opto = sum(ismember(success_trials, uopto_ep_trials));
                summary_eps(idx,2) = length(uopto_ep_trials) - success_opto; %opto_failure
                summary_eps(idx,4) = success_opto; %opto_success
                summary_eps(idx,6) = length(uopto_ep_trials); % opto#
            end
        end
    

        summary_eps(idx,7) = trialn; % total#
        failure_per_epoch = trialn - success_per_epoch;
        summary_eps(idx,3) = failure_per_epoch - summary_eps(idx,2); %nonopto_failure
        summary_eps(idx,5) = success_per_epoch - summary_eps(idx,4); %nonopto_success
         
    end
    
    if sum(summary_eps(:,6)) > 0
        day_n = string(['Opto ' char(day_cd)]);
    else
        day_n = day_cd;
    end
    summary_days(this_day,1) = sum(summary_eps(:,2)); %opto_failure
    summary_days(this_day,2) = sum(summary_eps(:,3)); %nonopto_failure
    summary_days(this_day,3) = sum(summary_eps(:,4)); %opto_success
    summary_days(this_day,4) = sum(summary_eps(:,5)); %nonopto_success
    summary_days(this_day,5) = sum(summary_eps(:,6)); %opto
    summary_days(this_day,6) = sum(summary_eps(:,7)); %total
    
    failure_percent_eps = zeros(length(rewloc),7); %opto_f/opto nonopto_f/nonopto_f opto_f/f nonopto_f/f opto_f/total nonopto_f/total f/total
    failure_percent_eps(:,1) = double(summary_eps(:,2))./double(summary_eps(:,6)); %opto_f/opto
    failure_percent_eps(:,2) = double(summary_eps(:,3))./(double(summary_eps(:,7))-double(summary_eps(:,6))); %nonopto_f/nonopto
    failure_percent_eps(:,3) = double(summary_eps(:,2))./(double(summary_eps(:,2))+double(summary_eps(:,3))); %opto_f/f
    failure_percent_eps(:,4) = double(summary_eps(:,3))./(double(summary_eps(:,2))+double(summary_eps(:,3))); %nonopto_f/f
    failure_percent_eps(:,5) = double(summary_eps(:,2))./double(summary_eps(:,7)); %opto_f/total
    failure_percent_eps(:,6) = double(summary_eps(:,3))./double(summary_eps(:,7)); %nonopto_f/total
    failure_percent_eps(:,7) = (double(summary_eps(:,2))+double(summary_eps(:,3)))./double(summary_eps(:,7)); %f/total
%     failure_percentage = double(total_failure_success_epochs(:,1))./double(total_failure_success_epochs(:,3));
    for r = 1:size(failure_percent_eps,1)
        for c = 1:size(failure_percent_eps,2)
            if isnan(failure_percent_eps(r,c))
                failure_percent_eps(r,c) = 0;
            end
        end
    end
    
    
    rewypos = l.changeRewLoc(rewloc)*1.5;
    all_epocvec(this_day,1:length(rewypos)) = 1:length(rewypos);
    all_rewypos(this_day, 1:length(rewypos)) = rewypos;
    epochs = [];
    for r1 = 1:size(summary_eps,1)
        if summary_eps(r1,6) > 0
            epochs = [epochs string(['ep' num2str(summary_eps(r1,1)) ' Opto'])];
        else
            epochs = [epochs string(['ep' num2str(summary_eps(r1,1))])];
        end
    end
   
    dates(this_day) = specific_date + " (" + char(day_n) + ")";
    f = figure('visible','off');
    eps = categorical(epochs);
    yyaxis left
    bg = bar(eps, summary_eps(:,2:5), 'stacked');
    colorss = lines(length(bg));
    for ii = 1:length(bg)
        set(bg(ii), 'FaceColor', colorss(ii,:),'FaceAlpha',0.85);
    end
    %text(eps, total_failure_success_epochs(:,1)./2,num2str(failure_percentage.*100, '%.1f')+"%",'Color','w','HorizontalAlignment','center','FontWeight','bold')
    ylabel("Number of failures and successes");
    hold on;
    %plot(eps, failure_percentage.*double(max(total_failure_success_epochs(:,3))),'LineWidth',2, 'Color','k','LineStyle','--');
    text(eps, summary_eps(:,2)-1,num2str(failure_percent_eps(:,5).*100, '%.1f')+"%",'Color','w','HorizontalAlignment','center','FontWeight','bold');
    text(eps, summary_eps(:,2)+summary_eps(:,3)./2,num2str(failure_percent_eps(:,6).*100, '%.1f')+"%",'Color','w','HorizontalAlignment','center','FontWeight','bold');
    text(eps, sum(summary_eps(:,2:3),2)+summary_eps(:,4)./2,num2str(summary_eps(:,4)./summary_eps(:,7).*100, '%.1f')+"%",'Color','w','HorizontalAlignment','center','FontWeight','bold');
    text(eps, sum(summary_eps(:,2:4),2)+summary_eps(:,5)./2,num2str(summary_eps(:,5)./summary_eps(:,7).*100, '%.1f')+"%",'Color','w','HorizontalAlignment','center','FontWeight','bold');
    yyaxis right
    scatter(eps, rewypos, 100,'k','filled','o');
    ylim([0 270]);
    ylabel("Reward location (cm)");
    title("Number of Failures During Each Epoch on "+day_cd+" ("+specific_date+"): "+mouse_cd);
    legend('Opto Failure', 'Non-Opto Failure', 'Opto Success', 'Non-Opto Success', 'Reward Location','color','none');
    %saveas(f,['H:\E186\E186\failure_fig\' char(day_cd) '_failureNum_w_opto.png'],'png')
    hold off;
end
disp('done!')
%% create failure plot for all days

failure_percent_days(:,1) = double(summary_days(:,1))./double(summary_days(:,5)); %opto_f/opto
failure_percent_days(:,2) = double(summary_days(:,2))./(double(summary_days(:,6))-double(summary_days(:,5))); %nonopto_f/nonopto
failure_percent_days(:,3) = double(summary_days(:,1))./(double(summary_days(:,1))+double(summary_days(:,2))); %opto_f/f
failure_percent_days(:,4) = double(summary_days(:,2))./(double(summary_days(:,1))+double(summary_days(:,2))); %nonopto_f/f
failure_percent_days(:,5) = double(summary_days(:,1))./double(summary_days(:,6)); %opto_f/total
failure_percent_days(:,6) = double(summary_days(:,2))./double(summary_days(:,6)); %nonopto_f/total
failure_percent_days(:,7) = (double(summary_days(:,1))+double(summary_days(:,2)))./double(summary_days(:,6)); %f/total
for r = 1:size(failure_percent_days,1)
    for c = 1:size(failure_percent_days,2)
        if isnan(failure_percent_days(r,c))
            failure_percent_days(r,c) = 0;
        end
    end
end

all_rewypos(all_rewypos==0) = NaN;
all_epocvec(all_epocvec==0) = NaN;
fig = figure('visible','off','Renderer','painters','Position', [20 20 2000 1500]);
x_dates = categorical(dates);
yyaxis left;
b = bar(x_dates, summary_days(:, 1:4), 'stacked');
colorss = lines(length(b));
for ii = 1:length(b)
    set(b(ii), 'FaceColor', colorss(ii,:),'FaceAlpha',0.85);
end
hold on;
%text(x_dates, total_failure_success_epochs(:,1)./2,num2str(failure_percentage.*100, '%.1f')+"%",'Color','w','HorizontalAlignment','center','FontWeight','bold')
title("Number of Failures per Imaging Day: "+mouse_cd);
%total_failure_success_days(:,3) = total_failure_success_days(:,1) + total_failure_success_days(:,2);
%freq = double(total_failure_success_days(:,1))./double(total_failure_success_days(:,3));
text(x_dates, summary_days(:,1)./2,num2str(failure_percent_days(:,5).*100, '%.0f')+"%",'Color','w','HorizontalAlignment','center','FontWeight','bold');
text(x_dates, summary_days(:,1)+summary_days(:,2)./2,num2str(failure_percent_days(:,6).*100, '%.0f')+"%",'Color','w','HorizontalAlignment','center','FontWeight','bold');
text(x_dates, sum(summary_days(:,1:2),2)+summary_days(:,3)./2,num2str(summary_days(:,3)./summary_days(:,6).*100, '%.0f')+"%",'Color','w','HorizontalAlignment','center','FontWeight','bold');
text(x_dates, sum(summary_days(:,1:3),2)+summary_days(:,4)./2./2,num2str(summary_days(:,4)./summary_days(:,6).*100, '%.0f')+"%",'Color','w','HorizontalAlignment','center','FontWeight','bold');
ylabel("Number of failures and successes");
yyaxis right;
scatter(x_dates, all_rewypos, 100,'k','o','LineWidth', 1.5);
hold on;
for iii = 1:size(all_epocvec,2)
    %scatter(x_dates, all_rewypos(:,iii), 100,'k','o','LineWidth', 1.5);hold on; %Matlab version issue
    text(x_dates, all_rewypos(:,iii), string(all_epocvec(:,iii)), 'Color','k','HorizontalAlignment','center','FontWeight','bold');
end
ylabel("Reward location (cm)");
ylim([0 270]);
legend('Opto Failure', 'Non-Opto Failure', 'Opto Success', 'Non-Opto Success', 'Reward Location','color','none');
%lgd.Location = 'northeastoutside';
saveas(fig,'H:\E186\E186\failure_fig\Total_failureNum_w_opto.png','png');
hold off;
disp("done!");