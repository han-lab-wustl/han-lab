load('Y:\E186\E186\alldays_info.mat')
F3_speed_3opto = [];
F3_speed_ctrl = [];
F3_nonstop_speed_3opto = [];
F3_nonstop_speed_ctrl = [];
F3_dark_speed_3opto = [];
F3_dark_speed_ctrl = [];
F3_13_speed_3opto = [];
F3_13_speed_ctrl = [];
all_speed_3opto = [];
all_nonstop_speed_3opto = [];
all_dark_speed_3opto = [];
all_13_speed_3opto = [];

F5_speed_5opto = [];
F5_speed_ctrl = [];
F5_nonstop_speed_5opto = [];
F5_nonstop_speed_ctrl = [];
F5_dark_speed_5opto = [];
F5_dark_speed_ctrl = [];
F5_13_speed_5opto = [];
F5_13_speed_ctrl = [];
all_speed_5opto = [];
all_nonstop_speed_5opto = [];
all_dark_speed_5opto = [];
all_13_speed_5opto = [];

all_speed_pbopto = [];
all_speed_ctrl = [];
all_nonstop_speed_pbopto = [];
all_nonstop_speed_ctrl = [];
all_dark_speed_pbopto = [];
all_dark_speed_ctrl = [];
all_13_speed_pbopto = [];
all_13_speed_ctrl = [];

rewloc_3opto = [];
rewloc_ctrl = [];
rewloc_pbopto = [];
rewloc_5opto = [];
%opto = [] %strings()
%first_3trials_licks = []
for this_day = 1:size(data,1)
    if data(this_day).Opto == 1 % only working on opto days here
        eps = data(this_day).day_eps;
        for i = 2:length([eps.epoch])-1
            avg_spd = eps(i).mean_speed;
            avg_spd_nonstop = eps(i).mean_speed_no_stop;
            avg_spd_dark = eps(i).mean_speed_in_dark;
            avg_spd_13 = eps(i).mean_speed_one_third_track;
            if sum(eps(i).opto_stim(1:3)) == 3 && sum(eps(i).probe_opto) == 2
                F3_speed_3opto = [F3_speed_3opto mean(avg_spd(1:3))];
                F3_nonstop_speed_3opto = [F3_nonstop_speed_3opto mean(avg_spd_nonstop(1:3))];
                F3_dark_speed_3opto = [F3_dark_speed_3opto mean(avg_spd_dark(1:3))];
                F3_13_speed_3opto = [F3_13_speed_3opto mean(avg_spd_13(1:3))];
                all_speed_3opto = [all_speed_3opto mean(avg_spd)];
                all_nonstop_speed_3opto = [all_nonstop_speed_3opto mean(avg_spd_nonstop)];
                all_dark_speed_3opto = [all_dark_speed_3opto mean(avg_spd_dark)];
                all_13_speed_3opto = [all_13_speed_3opto mean(avg_spd_13)];
                rewloc_3opto = [rewloc_3opto eps(i).RewLoc];
            elseif sum(eps(i).opto_stim(1:5)) == 0 && sum(eps(i).probe_opto) == 0
                F3_speed_ctrl = [F3_speed_ctrl mean(avg_spd(1:3))];
                F3_nonstop_speed_ctrl = [F3_nonstop_speed_ctrl mean(avg_spd_nonstop(1:3))];
                F3_dark_speed_ctrl = [F3_dark_speed_ctrl mean(avg_spd_dark(1:3))];
                F3_13_speed_ctrl = [F3_13_speed_ctrl mean(avg_spd_13(1:3))];
                F5_speed_ctrl = [F5_speed_ctrl mean(avg_spd(1:5))];
                F5_nonstop_speed_ctrl = [F5_nonstop_speed_ctrl mean(avg_spd_nonstop(1:5))];
                F5_dark_speed_ctrl = [F5_dark_speed_ctrl mean(avg_spd_dark(1:5))];
                F5_13_speed_ctrl = [F5_13_speed_ctrl mean(avg_spd_13(1:5))];
                all_speed_ctrl = [all_speed_ctrl mean(avg_spd)];
                all_nonstop_speed_ctrl = [all_nonstop_speed_ctrl mean(avg_spd_nonstop)];
                all_dark_speed_ctrl = [all_dark_speed_ctrl mean(avg_spd_dark)];
                all_13_speed_ctrl = [all_13_speed_ctrl mean(avg_spd_13)];
                rewloc_ctrl = [rewloc_ctrl eps(i).RewLoc];
            elseif sum(eps(i).opto_stim(1:3)) == 0 && sum(eps(i).probe_opto) == 2
                all_speed_pbopto = [all_speed_pbopto mean(avg_spd)];
                all_nonstop_speed_pbopto = [all_nonstop_speed_pbopto mean(avg_spd_nonstop)];
                all_dark_speed_pbopto = [all_dark_speed_pbopto mean(avg_spd_dark)];
                all_13_speed_pbopto = [all_13_speed_pbopto mean(avg_spd_13)];
                rewloc_pbopto = [rewloc_pbopto eps(i).RewLoc];
            elseif sum(eps(i).opto_stim(1:5)) == 5 && sum(eps(i).probe_opto) == 0
                F5_speed_5opto = [F5_speed_5opto mean(avg_spd(1:5))];
                F5_nonstop_speed_5opto = [F5_nonstop_speed_5opto mean(avg_spd_nonstop(1:5))];
                F5_dark_speed_5opto = [F5_dark_speed_5opto mean(avg_spd_dark(1:5))];
                F5_13_speed_5opto = [F5_13_speed_5opto mean(avg_spd_13(1:5))];
                all_speed_5opto = [all_speed_5opto mean(avg_spd)];
                all_nonstop_speed_5opto = [all_nonstop_speed_5opto mean(avg_spd_nonstop)];
                all_dark_speed_5opto = [all_dark_speed_5opto mean(avg_spd_dark)];
                all_13_speed_5opto = [all_13_speed_5opto mean(avg_spd_13)];
                rewloc_5opto = [rewloc_5opto eps(i).RewLoc];
            end
        end
%     else
%         eps = data(this_day).day_eps;
%         for i = 2:length([eps.epoch])-1
%             all_success = eps(i).success_info;
%             F3_speed_ct = [F3_speed_ct sum(all_success(1:3))/3*100];
%             F5_speed_ct = [F5_speed_ct sum(all_success(1:5))/5*100];
%             total_speed_ct = [total_speed_ct sum(all_success)/length(all_success)*100];
%             rewloc_ct = [rewloc_pbopto eps(i).RewLoc];
%         end
    end
end
%opto(strcmp(opto, '')) = []
%anovan(first_3trials_success_rate, {rewloc,opto}, 'model','interaction','varnames',{'RewLoc','Opto or not'})
[~, i3] = sort(rewloc_3opto);
rewloc_3opto = rewloc_3opto(i3);
F3_speed_3opto = F3_speed_3opto(i3);
F3_nonstop_speed_3opto = F3_nonstop_speed_3opto(i3);
F3_dark_speed_3opto = F3_dark_speed_3opto(i3);
F3_13_speed_3opto = F3_13_speed_3opto(i3);
all_speed_3opto = all_speed_3opto(i3);
all_nonstop_speed_3opto = all_nonstop_speed_3opto(i3);
all_dark_speed_3opto = all_dark_speed_3opto(i3);
all_13_speed_3opto = all_13_speed_3opto(i3);

[~, ic] = sort(rewloc_ctrl);
rewloc_ctrl = rewloc_ctrl(ic);
F3_speed_ctrl = F3_speed_ctrl(ic);
F3_nonstop_speed_ctrl = F3_nonstop_speed_ctrl(ic);
F3_dark_speed_ctrl = F3_dark_speed_ctrl(ic);
F3_13_speed_ctrl = F3_13_speed_ctrl(ic);
F5_speed_ctrl = F5_speed_ctrl(ic);
F5_nonstop_speed_ctrl = F5_nonstop_speed_ctrl(ic);
F5_dark_speed_ctrl = F5_dark_speed_ctrl(ic);
F5_13_speed_ctrl = F5_13_speed_ctrl(ic);
all_speed_ctrl = all_speed_ctrl(ic);
all_nonstop_speed_ctrl = all_nonstop_speed_ctrl(ic);
all_dark_speed_ctrl = all_dark_speed_ctrl(ic);
all_13_speed_ctrl = all_13_speed_ctrl(ic);

[~, ip] = sort(rewloc_pbopto);
rewloc_pbopto = rewloc_pbopto(ip);
all_speed_pbopto = all_speed_pbopto(ip);
all_nonstop_speed_pbopto = all_nonstop_speed_pbopto(ip);
all_dark_speed_pbopto = all_dark_speed_pbopto(ip);
all_13_speed_pbopto = all_13_speed_pbopto(ip);

[~, i5] = sort(rewloc_5opto);
rewloc_5opto = rewloc_5opto(i5);
F5_speed_5opto = F5_speed_5opto(i5);
F5_nonstop_speed_5opto = F5_nonstop_speed_5opto(i5);
F5_dark_speed_5opto = F5_dark_speed_5opto(i5);
F5_13_speed_5opto = F5_13_speed_5opto(i5);
all_speed_5opto = all_speed_5opto(i5);
all_nonstop_speed_5opto = all_nonstop_speed_5opto(i5);
all_dark_speed_5opto = all_dark_speed_5opto(i5);
all_13_speed_5opto = all_13_speed_5opto(i5);

% Reward zones: [67:86] [101:120] [135:154]
RZ1 = 67:86;
RZ2 = 101:120;
RZ3 = 135:154;

% First 3 trials w/ opto
F3zone11 = F3_speed_3opto(ismember(rewloc_3opto, RZ1));
F3zone21 = F3_speed_3opto(ismember(rewloc_3opto, RZ2));
F3zone31 = F3_speed_3opto(ismember(rewloc_3opto, RZ3));
F3zone12 = F3_nonstop_speed_3opto(ismember(rewloc_3opto, RZ1));
F3zone22 = F3_nonstop_speed_3opto(ismember(rewloc_3opto, RZ2));
F3zone32 = F3_nonstop_speed_3opto(ismember(rewloc_3opto, RZ3));
F3zone13 = F3_dark_speed_3opto(ismember(rewloc_3opto, RZ1));
F3zone23 = F3_dark_speed_3opto(ismember(rewloc_3opto, RZ2));
F3zone33 = F3_dark_speed_3opto(ismember(rewloc_3opto, RZ3));
F3zone14 = F3_13_speed_3opto(ismember(rewloc_3opto, RZ1));
F3zone24 = F3_13_speed_3opto(ismember(rewloc_3opto, RZ2));
F3zone34 = F3_13_speed_3opto(ismember(rewloc_3opto, RZ3));
F3all11 = all_speed_3opto(ismember(rewloc_3opto, RZ1));
F3all21 = all_speed_3opto(ismember(rewloc_3opto, RZ2));
F3all31 = all_speed_3opto(ismember(rewloc_3opto, RZ3));
F3all12 = all_nonstop_speed_3opto(ismember(rewloc_3opto, RZ1));
F3all22 = all_nonstop_speed_3opto(ismember(rewloc_3opto, RZ2));
F3all32 = all_nonstop_speed_3opto(ismember(rewloc_3opto, RZ3));
F3all13 = all_dark_speed_3opto(ismember(rewloc_3opto, RZ1));
F3all23 = all_dark_speed_3opto(ismember(rewloc_3opto, RZ2));
F3all33 = all_dark_speed_3opto(ismember(rewloc_3opto, RZ3));
F3all14 = all_13_speed_3opto(ismember(rewloc_3opto, RZ1));
F3all24 = all_13_speed_3opto(ismember(rewloc_3opto, RZ2));
F3all34 = all_13_speed_3opto(ismember(rewloc_3opto, RZ3));

% First 5 trials w/ opto
F5zone11 = F5_speed_5opto(ismember(rewloc_5opto, RZ1));
F5zone21 = F5_speed_5opto(ismember(rewloc_5opto, RZ2));
F5zone31 = F5_speed_5opto(ismember(rewloc_5opto, RZ3));
F5zone12 = F5_nonstop_speed_5opto(ismember(rewloc_5opto, RZ1));
F5zone22 = F5_nonstop_speed_5opto(ismember(rewloc_5opto, RZ2));
F5zone32 = F5_nonstop_speed_5opto(ismember(rewloc_5opto, RZ3));
F5zone13 = F5_dark_speed_5opto(ismember(rewloc_5opto, RZ1));
F5zone23 = F5_dark_speed_5opto(ismember(rewloc_5opto, RZ2));
F5zone33 = F5_dark_speed_5opto(ismember(rewloc_5opto, RZ3));
F5zone14 = F5_13_speed_5opto(ismember(rewloc_5opto, RZ1));
F5zone24 = F5_13_speed_5opto(ismember(rewloc_5opto, RZ2));
F5zone34 = F5_13_speed_5opto(ismember(rewloc_5opto, RZ3));
F5all11 = all_speed_5opto(ismember(rewloc_5opto, RZ1));
F5all21 = all_speed_5opto(ismember(rewloc_5opto, RZ2));
F5all31 = all_speed_5opto(ismember(rewloc_5opto, RZ3));
F5all12 = all_nonstop_speed_5opto(ismember(rewloc_5opto, RZ1));
F5all22 = all_nonstop_speed_5opto(ismember(rewloc_5opto, RZ2));
F5all32 = all_nonstop_speed_5opto(ismember(rewloc_5opto, RZ3));
F5all13 = all_dark_speed_5opto(ismember(rewloc_5opto, RZ1));
F5all23 = all_dark_speed_5opto(ismember(rewloc_5opto, RZ2));
F5all33 = all_dark_speed_5opto(ismember(rewloc_5opto, RZ3));
F5all14 = all_13_speed_5opto(ismember(rewloc_5opto, RZ1));
F5all24 = all_13_speed_5opto(ismember(rewloc_5opto, RZ2));
F5all34 = all_13_speed_5opto(ismember(rewloc_5opto, RZ3));

% Probes w/ opto
pbzone11 = all_speed_pbopto(ismember(rewloc_pbopto, RZ1));
pbzone21 = all_speed_pbopto(ismember(rewloc_pbopto, RZ2));
pbzone31 = all_speed_pbopto(ismember(rewloc_pbopto, RZ3));
pbzone12 = all_nonstop_speed_pbopto(ismember(rewloc_pbopto, RZ1));
pbzone22 = all_nonstop_speed_pbopto(ismember(rewloc_pbopto, RZ2));
pbzone32 = all_nonstop_speed_pbopto(ismember(rewloc_pbopto, RZ3));
pbzone13 = all_dark_speed_pbopto(ismember(rewloc_pbopto, RZ1));
pbzone23 = all_dark_speed_pbopto(ismember(rewloc_pbopto, RZ2));
pbzone33 = all_dark_speed_pbopto(ismember(rewloc_pbopto, RZ3));
pbzone14 = all_13_speed_pbopto(ismember(rewloc_pbopto, RZ1));
pbzone24 = all_13_speed_pbopto(ismember(rewloc_pbopto, RZ2));
pbzone34 = all_13_speed_pbopto(ismember(rewloc_pbopto, RZ3));

% ctrl trials
F3zone11_c = F3_speed_ctrl(ismember(rewloc_ctrl, RZ1));
F3zone21_c = F3_speed_ctrl(ismember(rewloc_ctrl, RZ2));
F3zone31_c = F3_speed_ctrl(ismember(rewloc_ctrl, RZ3));
F3zone12_c = F3_nonstop_speed_ctrl(ismember(rewloc_ctrl, RZ1));
F3zone22_c = F3_nonstop_speed_ctrl(ismember(rewloc_ctrl, RZ2));
F3zone32_c = F3_nonstop_speed_ctrl(ismember(rewloc_ctrl, RZ3));
F3zone13_c = F3_dark_speed_ctrl(ismember(rewloc_ctrl, RZ1));
F3zone23_c = F3_dark_speed_ctrl(ismember(rewloc_ctrl, RZ2));
F3zone33_c = F3_dark_speed_ctrl(ismember(rewloc_ctrl, RZ3));
F3zone14_c = F3_13_speed_ctrl(ismember(rewloc_ctrl, RZ1));
F3zone24_c = F3_13_speed_ctrl(ismember(rewloc_ctrl, RZ2));
F3zone34_c = F3_13_speed_ctrl(ismember(rewloc_ctrl, RZ3));

F5zone11_c = F5_speed_ctrl(ismember(rewloc_ctrl, RZ1));
F5zone21_c = F5_speed_ctrl(ismember(rewloc_ctrl, RZ2));
F5zone31_c = F5_speed_ctrl(ismember(rewloc_ctrl, RZ3));
F5zone12_c = F5_nonstop_speed_ctrl(ismember(rewloc_ctrl, RZ1));
F5zone22_c = F5_nonstop_speed_ctrl(ismember(rewloc_ctrl, RZ2));
F5zone32_c = F5_nonstop_speed_ctrl(ismember(rewloc_ctrl, RZ3));
F5zone13_c = F5_dark_speed_ctrl(ismember(rewloc_ctrl, RZ1));
F5zone23_c = F5_dark_speed_ctrl(ismember(rewloc_ctrl, RZ2));
F5zone33_c = F5_dark_speed_ctrl(ismember(rewloc_ctrl, RZ3));
F5zone14_c = F5_13_speed_ctrl(ismember(rewloc_ctrl, RZ1));
F5zone24_c = F5_13_speed_ctrl(ismember(rewloc_ctrl, RZ2));
F5zone34_c = F5_13_speed_ctrl(ismember(rewloc_ctrl, RZ3));

allzone11_c = all_speed_ctrl(ismember(rewloc_ctrl, RZ1));
allzone21_c = all_speed_ctrl(ismember(rewloc_ctrl, RZ2));
allzone31_c = all_speed_ctrl(ismember(rewloc_ctrl, RZ3));
allzone12_c = all_nonstop_speed_ctrl(ismember(rewloc_ctrl, RZ1));
allzone22_c = all_nonstop_speed_ctrl(ismember(rewloc_ctrl, RZ2));
allzone32_c = all_nonstop_speed_ctrl(ismember(rewloc_ctrl, RZ3));
allzone13_c = all_dark_speed_ctrl(ismember(rewloc_ctrl, RZ1));
allzone23_c = all_dark_speed_ctrl(ismember(rewloc_ctrl, RZ2));
allzone33_c = all_dark_speed_ctrl(ismember(rewloc_ctrl, RZ3));
allzone14_c = all_13_speed_ctrl(ismember(rewloc_ctrl, RZ1));
allzone24_c = all_13_speed_ctrl(ismember(rewloc_ctrl, RZ2));
allzone34_c = all_13_speed_ctrl(ismember(rewloc_ctrl, RZ3));

% First 3 trials w/ opto
figure('Renderer', 'painters', 'Position', [20 20 2000 1400]);
f1 = tiledlayout('flow');
nexttile
comparestatsplots(F3zone11,F3zone11_c,'b','r',[1 2]); hold on
comparestatsplots(F3zone21,F3zone21_c,'b','r',[4 5]);
comparestatsplots(F3zone31,F3zone31_c,'b','r',[7 8]);
xlim([0 9])
xticks([1 2 4 5 7 8])
xticklabels({'Opto: RZ1', 'Non-opto: RZ1', 'Opto: RZ2', 'Non-opto: RZ2', 'Opto: RZ3', 'Non-opto: RZ3'})
title({'Mean speed of first 3 trials',' '})
hold off
nexttile
comparestatsplots(F3zone12,F3zone12_c,'b','r',[1 2]); hold on
comparestatsplots(F3zone22,F3zone22_c,'b','r',[4 5]);
comparestatsplots(F3zone32,F3zone32_c,'b','r',[7 8]);
xlim([0 9])
xticks([1 2 4 5 7 8])
xticklabels({'Opto: RZ1', 'Non-opto: RZ1', 'Opto: RZ2', 'Non-opto: RZ2', 'Opto: RZ3', 'Non-opto: RZ3'})
title({'Mean speed (nonstop) of first 3 trials',' '})
hold off
nexttile
comparestatsplots(F3zone13,F3zone13_c,'b','r',[1 2]); hold on
comparestatsplots(F3zone23,F3zone23_c,'b','r',[4 5]);
comparestatsplots(F3zone33,F3zone33_c,'b','r',[7 8]);
xlim([0 9])
xticks([1 2 4 5 7 8])
xticklabels({'Opto: RZ1', 'Non-opto: RZ1', 'Opto: RZ2', 'Non-opto: RZ2', 'Opto: RZ3', 'Non-opto: RZ3'})
title({'Mean speed (in dark) of first 3 trials',' '})
hold off
nexttile
comparestatsplots(F3zone14,F3zone14_c,'b','r',[1 2]); hold on
comparestatsplots(F3zone24,F3zone24_c,'b','r',[4 5]);
comparestatsplots(F3zone34,F3zone34_c,'b','r',[7 8]);
xlim([0 9])
xticks([1 2 4 5 7 8])
xticklabels({'Opto: RZ1', 'Non-opto: RZ1', 'Opto: RZ2', 'Non-opto: RZ2', 'Opto: RZ3', 'Non-opto: RZ3'})
title({'Mean speed (1/3 track) of first 3 trials',' '})
hold off

% First 5 trials w/ opto
figure('Renderer', 'painters', 'Position', [20 20 2000 1400]);
f2 = tiledlayout('flow');
nexttile
comparestatsplots(F5zone11,F5zone11_c,'b','r',[1 2]); hold on
comparestatsplots(F5zone21,F5zone21_c,'b','r',[4 5]);
comparestatsplots(F5zone31,F5zone31_c,'b','r',[7 8]);
xlim([0 9])
xticks([1 2 4 5 7 8])
xticklabels({'Opto: RZ1', 'Non-opto: RZ1', 'Opto: RZ2', 'Non-opto: RZ2', 'Opto: RZ3', 'Non-opto: RZ3'})
title({'Mean speed of first 5 trials',' '})
hold off
nexttile
comparestatsplots(F5zone12,F5zone12_c,'b','r',[1 2]); hold on
comparestatsplots(F5zone22,F5zone22_c,'b','r',[4 5]);
comparestatsplots(F5zone32,F5zone32_c,'b','r',[7 8]);
xlim([0 9])
xticks([1 2 4 5 7 8])
xticklabels({'Opto: RZ1', 'Non-opto: RZ1', 'Opto: RZ2', 'Non-opto: RZ2', 'Opto: RZ3', 'Non-opto: RZ3'})
title({'Mean speed (nonstop) of first 5 trials',' '})
hold off
nexttile
comparestatsplots(F5zone13,F5zone13_c,'b','r',[1 2]); hold on
comparestatsplots(F5zone23,F5zone23_c,'b','r',[4 5]);
comparestatsplots(F5zone33,F5zone33_c,'b','r',[7 8]);
xlim([0 9])
xticks([1 2 4 5 7 8])
xticklabels({'Opto: RZ1', 'Non-opto: RZ1', 'Opto: RZ2', 'Non-opto: RZ2', 'Opto: RZ3', 'Non-opto: RZ3'})
title({'Mean speed (in dark) of first 5 trials',' '})
hold off
nexttile
comparestatsplots(F5zone14,F5zone14_c,'b','r',[1 2]); hold on
comparestatsplots(F5zone24,F5zone24_c,'b','r',[4 5]);
comparestatsplots(F5zone34,F5zone34_c,'b','r',[7 8]);
xlim([0 9])
xticks([1 2 4 5 7 8])
xticklabels({'Opto: RZ1', 'Non-opto: RZ1', 'Opto: RZ2', 'Non-opto: RZ2', 'Opto: RZ3', 'Non-opto: RZ3'})
title({'Mean speed (1/3 track) of first 5 trials',' '})
hold off

% Probes w/ opto
figure('Renderer', 'painters', 'Position', [20 20 2000 1400]);
f3 = tiledlayout('flow');
nexttile
comparestatsplots(pbzone11,allzone11_c,'b','r',[1 2]); hold on
comparestatsplots(pbzone21,allzone21_c,'b','r',[4 5]);
comparestatsplots(pbzone31,allzone31_c,'b','r',[7 8]);
xlim([0 9])
xticks([1 2 4 5 7 8])
xticklabels({'Opto: RZ1', 'Non-opto: RZ1', 'Opto: RZ2', 'Non-opto: RZ2', 'Opto: RZ3', 'Non-opto: RZ3'})
title({'Mean speed of epochs: probes w/ opto',' '})
hold off
nexttile
comparestatsplots(pbzone12,allzone12_c,'b','r',[1 2]); hold on
comparestatsplots(pbzone22,allzone22_c,'b','r',[4 5]);
comparestatsplots(pbzone32,allzone32_c,'b','r',[7 8]);
xlim([0 9])
xticks([1 2 4 5 7 8])
xticklabels({'Opto: RZ1', 'Non-opto: RZ1', 'Opto: RZ2', 'Non-opto: RZ2', 'Opto: RZ3', 'Non-opto: RZ3'})
title({'Mean speed (nonstop) of epochs: probes w/ opto',' '})
hold off
nexttile
comparestatsplots(pbzone13,allzone13_c,'b','r',[1 2]); hold on
comparestatsplots(pbzone23,allzone23_c,'b','r',[4 5]);
comparestatsplots(pbzone33,allzone33_c,'b','r',[7 8]);
xlim([0 9])
xticks([1 2 4 5 7 8])
xticklabels({'Opto: RZ1', 'Non-opto: RZ1', 'Opto: RZ2', 'Non-opto: RZ2', 'Opto: RZ3', 'Non-opto: RZ3'})
title({'Mean speed (in dark) of epochs: probes w/ opto',' '})
hold off
nexttile
comparestatsplots(pbzone14,allzone14_c,'b','r',[1 2]); hold on
comparestatsplots(pbzone24,allzone24_c,'b','r',[4 5]);
comparestatsplots(pbzone34,allzone34_c,'b','r',[7 8]);
xlim([0 9])
xticks([1 2 4 5 7 8])
xticklabels({'Opto: RZ1', 'Non-opto: RZ1', 'Opto: RZ2', 'Non-opto: RZ2', 'Opto: RZ3', 'Non-opto: RZ3'})
title({'Mean speed (1/3 track) of epochs: probes w/ opto',' '})
hold off

% Entire ep: First 3 trials have opto
figure('Renderer', 'painters', 'Position', [20 20 2000 1400]);
f4 = tiledlayout('flow');
nexttile
comparestatsplots(F3all11,allzone11_c,'b','r',[1 2]); hold on
comparestatsplots(F3all21,allzone21_c,'b','r',[4 5]);
comparestatsplots(F3all31,allzone31_c,'b','r',[7 8]);
xlim([0 9])
xticks([1 2 4 5 7 8])
xticklabels({'Opto: RZ1', 'Non-opto: RZ1', 'Opto: RZ2', 'Non-opto: RZ2', 'Opto: RZ3', 'Non-opto: RZ3'})
title({'Mean speed of epochs: first 3 trials w/ opto',' '})
hold off
nexttile
comparestatsplots(F3all12,allzone12_c,'b','r',[1 2]); hold on
comparestatsplots(F3all22,allzone22_c,'b','r',[4 5]);
comparestatsplots(F3all32,allzone32_c,'b','r',[7 8]);
xlim([0 9])
xticks([1 2 4 5 7 8])
xticklabels({'Opto: RZ1', 'Non-opto: RZ1', 'Opto: RZ2', 'Non-opto: RZ2', 'Opto: RZ3', 'Non-opto: RZ3'})
title({'Mean speed (nonstop) of epochs: first 3 trials w/ opto',' '})
hold off
nexttile
comparestatsplots(F3all13,allzone13_c,'b','r',[1 2]); hold on
comparestatsplots(F3all23,allzone23_c,'b','r',[4 5]);
comparestatsplots(F3all33,allzone33_c,'b','r',[7 8]);
xlim([0 9])
xticks([1 2 4 5 7 8])
xticklabels({'Opto: RZ1', 'Non-opto: RZ1', 'Opto: RZ2', 'Non-opto: RZ2', 'Opto: RZ3', 'Non-opto: RZ3'})
title({'Mean speed (in dark) of epochs: first 3 trials w/ opto',' '})
hold off
nexttile
comparestatsplots(F3all14,allzone14_c,'b','r',[1 2]); hold on
comparestatsplots(F3all24,allzone24_c,'b','r',[4 5]);
comparestatsplots(F3all34,allzone34_c,'b','r',[7 8]);
xlim([0 9])
xticks([1 2 4 5 7 8])
xticklabels({'Opto: RZ1', 'Non-opto: RZ1', 'Opto: RZ2', 'Non-opto: RZ2', 'Opto: RZ3', 'Non-opto: RZ3'})
title({'Mean speed (1/3 track) of epochs: first 3 trials w/ opto',' '})
hold off

% Entire ep: First 5 trials have opto
figure('Renderer', 'painters', 'Position', [20 20 2000 1400]);
f5 = tiledlayout('flow');
nexttile
comparestatsplots(F5all11,allzone11_c,'b','r',[1 2]); hold on
comparestatsplots(F5all21,allzone21_c,'b','r',[4 5]);
comparestatsplots(F5all31,allzone31_c,'b','r',[7 8]);
xlim([0 9])
xticks([1 2 4 5 7 8])
xticklabels({'Opto: RZ1', 'Non-opto: RZ1', 'Opto: RZ2', 'Non-opto: RZ2', 'Opto: RZ3', 'Non-opto: RZ3'})
title({'Mean speed of epochs: first 5 trials w/ opto',' '})
hold off
nexttile
comparestatsplots(F5all12,allzone12_c,'b','r',[1 2]); hold on
comparestatsplots(F5all22,allzone22_c,'b','r',[4 5]);
comparestatsplots(F5all32,allzone32_c,'b','r',[7 8]);
xlim([0 9])
xticks([1 2 4 5 7 8])
xticklabels({'Opto: RZ1', 'Non-opto: RZ1', 'Opto: RZ2', 'Non-opto: RZ2', 'Opto: RZ3', 'Non-opto: RZ3'})
title({'Mean speed (nonstop) of epochs: first 5 trials w/ opto',' '})
hold off
nexttile
comparestatsplots(F5all13,allzone13_c,'b','r',[1 2]); hold on
comparestatsplots(F5all23,allzone23_c,'b','r',[4 5]);
comparestatsplots(F5all33,allzone33_c,'b','r',[7 8]);
xlim([0 9])
xticks([1 2 4 5 7 8])
xticklabels({'Opto: RZ1', 'Non-opto: RZ1', 'Opto: RZ2', 'Non-opto: RZ2', 'Opto: RZ3', 'Non-opto: RZ3'})
title({'Mean speed (in dark) of epochs: first 5 trials w/ opto',' '})
hold off
nexttile
comparestatsplots(F5all14,allzone14_c,'b','r',[1 2]); hold on
comparestatsplots(F5all24,allzone24_c,'b','r',[4 5]);
comparestatsplots(F5all34,allzone34_c,'b','r',[7 8]);
xlim([0 9])
xticks([1 2 4 5 7 8])
xticklabels({'Opto: RZ1', 'Non-opto: RZ1', 'Opto: RZ2', 'Non-opto: RZ2', 'Opto: RZ3', 'Non-opto: RZ3'})
title({'Mean speed (1/3 track) of epochs: first 5 trials w/ opto',' '})
hold off

% All rewardzones
figure('Renderer', 'painters', 'Position', [20 20 2000 1400])
f6 = tiledlayout('flow');
nexttile
comparestatsplots(F3_speed_3opto,F3_speed_ctrl,'b','r',[1 2]); hold on
comparestatsplots(all_speed_3opto,all_speed_ctrl,'b','r',[4 5]);
comparestatsplots(F5_speed_5opto,F5_speed_ctrl,'b','r',[7 8]);
comparestatsplots(all_speed_5opto,all_speed_ctrl,'b','r',[10 11]);
comparestatsplots(all_speed_pbopto,all_speed_ctrl,'b','r',[13 14]);
xlim([0 15])
xticks([1 2 4 5 7 8 10 11 13 14])
xticklabels({'F3Opto: F3', 'F3Non-opto: F3', 'F3Opto: all', 'F3Non-opto: all', 'F5Opto: F5', 'F5Non-opto: F5', 'F5Opto: all', 'F5Non-opto: all', 'pbOpto: all', 'pbNon-opto: all'})
title({'Mean speed of all conditions',' '})
hold off

nexttile
comparestatsplots(F3_nonstop_speed_3opto,F3_nonstop_speed_ctrl,'b','r',[1 2]); hold on
comparestatsplots(all_nonstop_speed_3opto,all_nonstop_speed_ctrl,'b','r',[4 5]);
comparestatsplots(F5_nonstop_speed_5opto,F5_nonstop_speed_ctrl,'b','r',[7 8]);
comparestatsplots(all_nonstop_speed_5opto,all_nonstop_speed_ctrl,'b','r',[10 11]);
comparestatsplots(all_nonstop_speed_pbopto,all_nonstop_speed_ctrl,'b','r',[13 14]);
xlim([0 15])
xticks([1 2 4 5 7 8 10 11 13 14])
xticklabels({'F3Opto: F3', 'F3Non-opto: F3', 'F3Opto: all', 'F3Non-opto: all', 'F5Opto: F5', 'F5Non-opto: F5', 'F5Opto: all', 'F5Non-opto: all', 'pbOpto: all', 'pbNon-opto: all'})
title({'Mean speed (nonstop) of all conditions',' '})
hold off

nexttile
comparestatsplots(F3_dark_speed_3opto,F3_dark_speed_ctrl,'b','r',[1 2]); hold on
comparestatsplots(all_dark_speed_3opto,all_dark_speed_ctrl,'b','r',[4 5]);
comparestatsplots(F5_dark_speed_5opto,F5_dark_speed_ctrl,'b','r',[7 8]);
comparestatsplots(all_dark_speed_5opto,all_dark_speed_ctrl,'b','r',[10 11]);
comparestatsplots(all_dark_speed_pbopto,all_dark_speed_ctrl,'b','r',[13 14]);
xlim([0 15])
xticks([1 2 4 5 7 8 10 11 13 14])
xticklabels({'F3Opto: F3', 'F3Non-opto: F3', 'F3Opto: all', 'F3Non-opto: all', 'F5Opto: F5', 'F5Non-opto: F5', 'F5Opto: all', 'F5Non-opto: all', 'pbOpto: all', 'pbNon-opto: all'})
title({'Mean speed (in dark) of all conditions',' '})
hold off

nexttile
comparestatsplots(F3_13_speed_3opto,F3_13_speed_ctrl,'b','r',[1 2]); hold on
comparestatsplots(all_13_speed_3opto,all_13_speed_ctrl,'b','r',[4 5]);
comparestatsplots(F5_13_speed_5opto,F5_13_speed_ctrl,'b','r',[7 8]);
comparestatsplots(all_13_speed_5opto,all_13_speed_ctrl,'b','r',[10 11]);
comparestatsplots(all_13_speed_pbopto,all_13_speed_ctrl,'b','r',[13 14]);
xlim([0 15])
xticks([1 2 4 5 7 8 10 11 13 14])
xticklabels({'F3Opto: F3', 'F3Non-opto: F3', 'F3Opto: all', 'F3Non-opto: all', 'F5Opto: F5', 'F5Non-opto: F5', 'F5Opto: all', 'F5Non-opto: all', 'pbOpto: all', 'pbNon-opto: all'})
title({'Mean speed (1/3 track) of all conditions',' '})
hold off

saveas(f1,'H:\E186\E186\speed\F3_1.png','png')
saveas(f2,'H:\E186\E186\speed\F5_1.png','png')
saveas(f3,'H:\E186\E186\speed\pb_1.png','png')
saveas(f4,'H:\E186\E186\speed\F3all_1.png','png')
saveas(f5,'H:\E186\E186\speed\F5all_1.png','png')
saveas(f6,'H:\E186\E186\speed\allall_1.png','png')

disp('Done!')