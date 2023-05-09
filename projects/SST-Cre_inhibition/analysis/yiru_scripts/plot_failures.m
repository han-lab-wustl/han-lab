%% Plot failures
% Could write another one that uses 'alldays_info.mat'

% Adjustable: invisible figures for single day or all days, save figures
% for single day or all days.

% The y position is scaled by multiplying 1.5 for ybinned and
% reward location; reward range was changed too.

Settings.paths = dir('Y:\E186\E186\D*\Fall.mat');
Settings.level_mouse_name = 3;
Settings.level_day = 4;
total_failure_success_days = int16.empty;
dates = strings(1, size(Settings.paths,1));
all_rewypos = zeros(size(Settings.paths,1),4);
all_epocvec = zeros(size(Settings.paths,1),4);

% Create faliure plot for each day
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
    
    rewrange = 10 *2/3; %rewardzonesize = 10
    rewloc = find(l.changeRewLoc ~= 0);
    total_failure_success_epochs = zeros(length(rewloc), 3);
    loc = [rewloc, size(l.changeRewLoc, 2)];
    for idx = 1:length(rewloc)
        rewl = l.changeRewLoc(1, loc(idx));
        rl = find(l.ybinned > rewl-rewrange & l.ybinned < rewl+rewrange & l.rewards == 1); % enlarge the range a little bit
        rl = rl(rl < loc(idx+1) & rl >= loc(idx));
        success_per_epoch = length(unique(l.trialnum(rl)));
        single_epoch = l.trialnum(loc(idx):loc(idx+1)-1);
        valid_trials = [2 single_epoch(single_epoch > 2)];
        trialn= valid_trials(end)-2;
        trialmax = valid_trials(end);
        %rewtrialmax= max(l.trialnum(rl));
        last_trial = find(l.trialnum == trialmax);% & l.ybinned ~= 1);
        last_trial = last_trial(ismember(last_trial, loc(idx)+1:loc(idx+1)));
        if length(unique(l.ybinned(last_trial))) == 1 % && ~isempty(last_trial)
            trialn = trialn-1;
        elseif max(l.ybinned(last_trial))<rewl-rewrange/2%~isempty(last_trial) && sum(l.ybinned(last_trial)>=rewl-rewrange/2)<1 % remove the last one without reaching the reward zone
                trialn = trialn-1;
        end

        failure_per_epoch = trialn - success_per_epoch;
        total_failure_success_epochs(idx,1) = failure_per_epoch;
        total_failure_success_epochs(idx,2) = success_per_epoch;
        total_failure_success_epochs(idx,3) = trialn; %failure_per_epoch + success_per_epoch;
    end
    failure_per_day = sum(total_failure_success_epochs(:,1));
    success_per_day = sum(total_failure_success_epochs(:,2));
    failure_percentage = double(total_failure_success_epochs(:,1))./double(total_failure_success_epochs(:,3));
    for p = 1:length(failure_percentage)
        if isnan(failure_percentage(p))
            failure_percentage(p) = 0;
        end
    end
    rewypos = l.changeRewLoc(rewloc)*1.5;
    all_epocvec(this_day,1:length(rewypos)) = 1:length(rewypos);
    all_rewypos(this_day, 1:length(rewypos)) = rewypos;
    epochs = "ep" + string(1: length(rewloc));

    f = figure('visible','off');
    eps = categorical(epochs);
    yyaxis left
    bg = bar(eps, total_failure_success_epochs(:,1:2), 'stacked');
    colorss = lines(length(bg));
    for ii = 1:length(bg)
        set(bg(ii), 'FaceColor', colorss(ii,:),'FaceAlpha',0.85);
    end
    ylabel("Number of failures and successes");
    hold on;
    text(eps, total_failure_success_epochs(:,1)./2,num2str(failure_percentage.*100, '%.1f')+"%",'Color','w','HorizontalAlignment','center','FontWeight','bold');
    yyaxis right
    scatter(eps, rewypos, 100,'k','filled','o');
    ylim([0 270]);
    ylabel("Reward location (cm)");
    title("Number of Failures During Each Epoch on "+ day_cd+ " ("+ specific_date+ "): "+mouse_cd);
    legend('Failure', 'Success', 'Reward Location','color','none');
    total_failure_success_days(this_day, :) = [failure_per_day success_per_day];
    %saveas(f,['H:\E186\E186\failure_fig\' char(day_cd) '_failureNum.png'],'png')
    hold off;
end
disp('done!')
%% create failure plot for all days
all_rewypos(all_rewypos==0) = NaN;
all_epocvec(all_epocvec==0) = NaN;
fig = figure('visible','off','Renderer','painters','Position', [20 20 2000 1500]);
x_dates = categorical(dates);
yyaxis left;
b = bar(x_dates, total_failure_success_days(:, 1:2), 'stacked');
colorss = lines(length(b));
for ii = 1:length(b)
    set(b(ii), 'FaceColor', colorss(ii,:),'FaceAlpha',0.85);
end
hold on;

title("Number of Failures per Imaging Day: "+mouse_cd);
total_failure_success_days(:,3) = total_failure_success_days(:,1) + total_failure_success_days(:,2);
freq = double(total_failure_success_days(:,1))./double(total_failure_success_days(:,3));
text(x_dates, double(total_failure_success_days(:,1))./2,num2str(freq.*100, '%.0f')+"%",'Color','w','HorizontalAlignment','center');
ylabel("Number of failures and successes");
yyaxis right;
scatter(x_dates, all_rewypos, 100,'k','o','LineWidth', 1.5);
hold on;
for iii = 1:size(all_epocvec,2)
    text(x_dates, all_rewypos(:,iii), string(all_epocvec(:,iii)), 'Color','k','HorizontalAlignment','center','FontWeight','bold');
end
ylabel("Reward location (cm)");
ylim([0 270]);
legend('Failure', 'Success', 'Reward Location','color','none');
%lgd.Location = 'northeastoutside';
saveas(fig,'H:\E186\E186\failure_fig\Total_failureNum.png','png');
hold off;
disp("done!");