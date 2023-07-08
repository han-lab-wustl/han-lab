% Zahra
% ep 2 days
% there is a bug in trialnum from VRalign/startend - ask Gerardo
% above bug is fixed in commit from 6/12/23
clear all; clear all;
% srcdir = 'Y:\sstcre_imaging';
% animal = 'e200';
srcdir = 'Z:\sstcre_imaging';
animal = 'e201';

grayColor = [.7 .7 .7];
days = [55 58 61 64 67 70 73 77 80 83 86 88 91 92];
plot_dff = 1; %plots the average dff of all cells across the session with the opto stim window
% days = [65 68 71 74 78 81 84 87];
[dffs,diff_ep2,spatial_info_ep2] = collect_neural_data_opto(days,srcdir,animal,2,plot_dff);

% ep 3
days = [56 59 62 65 68 71 75 78 82 84 87 89];
% days = [66 69 72 75 79 82 88];
[dffs,diff_ep3,spatial_info_ep3] = collect_neural_data_opto(days,srcdir,animal,3,plot_dff);
% compare to control ep 2 and 3
% days = [57 60 63 66 69 72 76 79 85 90];
days = [67 70 73 76 80 83 86 89];
[diff_ctrl_ep2,spatial_info_ctrl_ep2] = collect_neural_data_opto(days,srcdir,animal,2);
[diff_ctrl_ep3,spatial_info_ctrl_ep3] = collect_neural_data_opto(days,srcdir,animal,3);
%%
% plot spatial info as a function of diff
figure; 
si = cell2mat(spatial_info_ep3);
si_ = si(diff_ep3>0.01);
[b,ind] = sort(si_);
plot(diff_ep3(diff_ep3>0.01), si_,'ro')
xlim([-0.1 0.3])
% ylim([0 1.4])
xlabel('ep3, first 5 trials (opto) - led off ep 3')
ylabel('ranked spatial info')
%%

% exclude cells diff < -4 --> is this ok to do??? find which cells these
% are
ex_diff_opto = diff_op(diff_opto_ep3_ep1>-4);
ex_diff_ctrl = diff_ctrl_ep3_ep1(diff_ctrl_ep3_ep1<10); % exclude cell with diff 15 --> whhich cell?
[h,p,~,stat] = ttest2(ex_diff_opto,ex_diff_ctrl)


figure; plot(1, ex_diff_opto, 'ro'); hold on; plot(2, ex_diff_ctrl, 'ko');
plot(1, mean(ex_diff_opto), 'r_', 'MarkerSize',30, 'LineWidth',3)
plot(2, mean(ex_diff_ctrl,'omitnan'), 'k_', 'MarkerSize',30, 'LineWidth',3)
xlim([0 3])
xticks([1 2])
xticklabels([{'ep3, first 5 trials (opto) - led off ep 3'}, {'ep3, first 5 trials (ctrl) - led off ep 3'}])
title(sprintf('%s, opto epoch 3', animal))