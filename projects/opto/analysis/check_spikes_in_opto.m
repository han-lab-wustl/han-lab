% Zahra
% ep 2 days
% there is a bug in trialnum from VRalign/startend - ask Gerardo
% above bug is fixed in commit from 6/12/23
clear all; clear all; close all;
% srcdir = 'Y:\sstcre_imaging';
% animal = 'e200';
srcdir = 'Z:\sstcre_imaging';
animal = 'e201';

grayColor = [.7 .7 .7];
days = [55 58 61 64 67];% 70 73 77];
plot_dff = 0; %plots the average dff of all cells across the session with the opto stim window
% days = [65 68 71 74 78 81 84 87];

[diff_opto,spatial_info,spike_optos,spike_opto_comps,...
    spiketime_av_optos, spiketime_av_opto_comps]= collect_neural_data_opto(days, ...
    srcdir,animal, 2,5,plot_dff);

% ep 3
% days = [56 62 65 68];
% % days = [66 69 72 75 79 82 88];
% [dffs,diff_opto_ep3,spatial_info_ep3, spike_opto_ep3, ...
%     spike_opto_comps_ep3] = collect_neural_data_opto(days, ...
%     srcdir,animal,3,5,plot_dff);
% compare to control ep 2 and 3
days = [57 60 63 66 69];
% days = [67 70 73 76 80 83 86 89];
[diff_opto_ctrl,spatial_info_ctrl,spike_optos_ctrl,spike_opto_comps_ctrl,...
    spiketime_av_optos_ctrl, spiketime_av_opto_comps_ctrl] = collect_neural_data_opto(days, ...
    srcdir,animal,2,5, plot_dff);
% [diff_ctrl_ep3,spatial_info_ctrl_ep3] = collect_neural_data_opto(days, ...
%     srcdir,animal,3,5, ...
%     plot_dff);
%%
% plot spatial info as a function of diff
figure; 
si = cell2mat(spatial_info);
si_ = si(diff_opto>0.01);
[b,ind] = sort(si_);
plot(diff_opto(diff_opto>0.01), si_,'ro')
xlim([-0.1 0.3])
% ylim([0 1.4])
xlabel('ep2, first 5 trials (opto) - led off ep 2')
ylabel('ranked spatial info')
%%

% exclude cells diff < -4 --> is this ok to do??? find which cells these
% are
ex_diff_opto = diff_opto(diff_opto>-4);
% ex_diff_ctrl = diff_ctrl_ep3_ep1(diff_ctrl_ep3_ep1<10); % exclude cell with diff 15 --> whhich cell?
[h,p,~,stat] = ttest2(ex_diff_opto,diff_opto_ctrl)
%%
dy = 2;
diff_spike_opto = diff([spiketime_av_optos{dy}; spiketime_av_opto_comps{dy}]);
diff_spike_opto_ctrl = diff([spiketime_av_optos_ctrl{dy}; spiketime_av_opto_comps_ctrl{dy}]);
[h,p,~,stat] =  ttest2(diff_spike_opto,diff_spike_opto_ctrl)

figure; plot(1, ex_diff_opto, 'ro'); hold on; plot(2, diff_opto_ctrl, 'ko');
plot(1, mean(ex_diff_opto), 'r_', 'MarkerSize',30, 'LineWidth',3)
plot(2, mean(diff_opto_ctrl,'omitnan'), 'k_', 'MarkerSize',30, 'LineWidth',3)
xlim([0 3])
xticks([1 2])
xticklabels([{'ep2, first 5 trials (opto) - led off ep 2'}, {'ep2, first 5 trials (ctrl) - led off ep 2'}])
title(sprintf('%s, opto epoch 2', animal))