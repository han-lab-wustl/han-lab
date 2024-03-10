% celltrack quantification figure
% figure 2 a/b
clear all
grayColor = [.7 .7 .7];
per_day_tracking = [124 436 202 0]; % e218. e145 e201 e186
week2week_tracking = [979 1288 1308 293];
per_day_tracking_activeonceweek = [1709 2006 1866 331];
figure('Renderer','painters')
bar([mean(per_day_tracking) mean(week2week_tracking) mean(per_day_tracking_activeonceweek)], 'FaceColor', 'w'); hold on
plot(1, per_day_tracking, 'ko'); plot(2, week2week_tracking, 'ko');
plot(3, per_day_tracking_activeonceweek, 'ko');
for i=1:length(per_day_tracking)
    plot([1 2 3], [per_day_tracking(i) week2week_tracking(i) per_day_tracking_activeonceweek(i)], 'k-')
end
ylabel('Tracked Cells')
xticklabels({"CellReg Daily Tracking", "CellReg + Weekly Concatenation", "CellReg Conditional"})
xtickangle(45)
box off
