function p = sr_comparisonplots_recentdays(opto_data, ctrl_data, pre_color, post_color, labels)
nexttile
comparestatsplots(opto_data(:,3), ctrl_data(:,3), pre_color, post_color, [1 2]); hold on
xlim([0 3])
xticks([1 2])
ylim([0 1])
%text(0.5,0.95, ['n = ', num2str(length(opto_data(:, 3)))])
%text(2.2,0.95, ['n = ', num2str(length(ctrl_data(:, 3)))])
ylabel('Success Rate')
xticklabels({labels(1), labels(2)})
title('Epochs with opto vs. non-opto')
hold off

nexttile
comparestatsplots(opto_data(opto_data(:,4)==1,3),ctrl_data(ctrl_data(:,4)<35,3), pre_color, post_color, [1 2]); hold on
xlim([0 3])
xticks([1 2])
ylabel('Success Rate')
ylim([0 1])
xticklabels({labels(3), labels(4)})
title('Opto in 2 probe trials vs. non-opto')
%text(0.5,0.95, ['n = ', num2str(length(opto_data(opto_data(:,4)==1,3)))])
%text(2.2,0.95, ['n = ', num2str(length(ctrl_data(ctrl_data(:,4)<35,3)))])
hold off

nexttile
comparestatsplots(opto_data(opto_data(:,4)==2,3),ctrl_data(ismember(ctrl_data(:,4),34:43),3), pre_color, post_color, [1 2]); hold on
xlim([0 3])
xticks([1 2])
ylabel('Success Rate')
ylim([0 1])
xticklabels({labels(5), labels(6)})
title('Opto in 2 probe trials + 3 epoch trials vs. non-opto')
%%text(0.5,0.95, ['n = ', num2str(length(opto_data(opto_data(:,4)==2,3)))])
%%text(2.2,0.95, ['n = ', num2str(length(ctrl_data(ismember(ctrl_data(:,4),34:43),3)))])
hold off

nexttile
comparestatsplots(opto_data(opto_data(:,4)==3,3),ctrl_data(ctrl_data(:,4)>51,3), pre_color, post_color, [1 2]); hold on
xlim([0 3])
xticks([1 2])
ylim([0 1])
ylabel('Success Rate')
xticklabels({labels(7), labels(8)})
title('Opto in 5 epoch trials vs. non-opto')
%text(0.5,0.95, ['n = ', num2str(length(opto_data(opto_data(:,4)==3,3)))])
%text(2.2,0.95, ['n = ', num2str(length(ctrl_data(ctrl_data(:,4)>51,3)))])
hold off
end
