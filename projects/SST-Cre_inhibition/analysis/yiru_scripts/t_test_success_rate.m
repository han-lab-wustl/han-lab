load('Y:\E186\E186\alldays_info.mat')
F3trials_SR_3opto = [];
F3trials_SR_ctrl = [];
F5trials_SR_ctrl = [];
F3trials_SR_pbopto = [];
F5trials_SR_pbopto = [];
F5trials_SR_5opto = [];
F3trials_SR_ct = [];
F5trials_SR_ct = [];
total_SR_3opto = [];
total_SR_ctrl = [];
total_SR_pbopto = [];
total_SR_5opto = [];
total_SR_ct = [];
rewloc_3opto = [];
rewloc_ctrl = [];
rewloc_pbopto = [];
rewloc_5opto = [];
rewloc_ct = [];
%opto = [] %strings()
%first_3trials_licks = []
for this_day = 1:size(data,1)
    if data(this_day).Opto == 1 % only working on opto days here
        eps = data(this_day).day_eps;
        for i = 2:length([eps.epoch])-1
            all_success = eps(i).success_info;
            if sum(eps(i).opto_stim(1:3)) == 3 && sum(eps(i).probe_opto) == 2
                F3trials_SR_3opto = [F3trials_SR_3opto sum(all_success(1:3))/3];
                total_SR_3opto = [total_SR_3opto sum(all_success)/length(all_success)];
                rewloc_3opto = [rewloc_3opto eps(i).RewLoc];
            elseif sum(eps(i).opto_stim(1:5)) == 0 && sum(eps(i).probe_opto) == 0
                F3trials_SR_ctrl = [F3trials_SR_ctrl sum(all_success(1:3))/3];
                F5trials_SR_ctrl = [F5trials_SR_ctrl sum(all_success(1:5))/5];
                total_SR_ctrl = [total_SR_ctrl sum(all_success)/length(all_success)];
                rewloc_ctrl = [rewloc_ctrl eps(i).RewLoc];
            elseif sum(eps(i).opto_stim(1:3)) == 0 && sum(eps(i).probe_opto) == 2
                F3trials_SR_pbopto = [F3trials_SR_pbopto sum(all_success(1:3))/3];
                F5trials_SR_pbopto = [F5trials_SR_pbopto sum(all_success(1:5))/5]; % maybe I don't need these two just the total SR
                total_SR_pbopto = [total_SR_pbopto sum(all_success)/length(all_success)];
                rewloc_pbopto = [rewloc_pbopto eps(i).RewLoc];
            elseif sum(eps(i).opto_stim(1:5)) == 5 && sum(eps(i).probe_opto) == 0
                F5trials_SR_5opto = [F5trials_SR_5opto sum(all_success(1:5))/5];
                total_SR_5opto = [total_SR_5opto sum(all_success)/length(all_success)];
                rewloc_5opto = [rewloc_5opto eps(i).RewLoc];
            end
        end
    else
        eps = data(this_day).day_eps;
        for i = 2:length([eps.epoch])-1
            all_success = eps(i).success_info;
            F3trials_SR_ct = [F3trials_SR_ct sum(all_success(1:3))/3];
            F5trials_SR_ct = [F5trials_SR_ct sum(all_success(1:5))/5];
            total_SR_ct = [total_SR_ct sum(all_success)/length(all_success)];
            rewloc_ct = [rewloc_pbopto eps(i).RewLoc];
        end
    end
end

f0 = figure('Renderer','painters','Position', [20 20 1000 700]);
comparestatsplots(F3trials_SR_3opto, F3trials_SR_ctrl,'b','r',[1 2]);hold on
comparestatsplots(total_SR_3opto, total_SR_ctrl,'b','r',[4 5])
comparestatsplots(F5trials_SR_5opto, F5trials_SR_ctrl,'b','r',[7 8])
comparestatsplots(total_SR_5opto, total_SR_ctrl,'b','r',[10 11])
comparestatsplots(total_SR_pbopto, total_SR_ctrl,'b','r',[13 14])
xlim([0 15])
xticks([1 2 4 5 7 8 10 11 13 14])
xticklabels({'Opto: F3', 'Non-opto: F3', 'Opto: F3all', 'Non-opto: F3all', 'Opto: F5', 'Non-opto: F5', 'Opto: F5all', 'Non-opto: F5all', 'Opto: Probe', 'Non-opto: Probe'})
title({'Success rates of all conditions',' '})
ylabel('Success rate')
ylim([0 1])
hold off
saveas(f0,'H:\E186\E186\success_rate\all_1.png','png')

%%
[~, i3] = sort(rewloc_3opto);
rewloc_3opto = rewloc_3opto(i3);
F3trials_SR_3opto = F3trials_SR_3opto(i3);
total_SR_3opto = total_SR_3opto(i3);

[~, ic] = sort(rewloc_ctrl);
rewloc_ctrl = rewloc_ctrl(ic);
F3trials_SR_ctrl = F3trials_SR_ctrl(ic);
F5trials_SR_ctrl = F5trials_SR_ctrl(ic);
total_SR_ctrl = total_SR_ctrl(ic);

[~, ip] = sort(rewloc_pbopto);
rewloc_pbopto = rewloc_pbopto(ip);
F3trials_SR_pbopto = F3trials_SR_pbopto(ip);
F5trials_SR_pbopto = F5trials_SR_pbopto(ip);
total_SR_pbopto = total_SR_pbopto(ip);

[~, i5] = sort(rewloc_5opto);
rewloc_5opto = rewloc_5opto(i5);
F5trials_SR_5opto = F5trials_SR_5opto(i5);
total_SR_5opto = total_SR_5opto(i5);

[~, ict] = sort(rewloc_ct);
rewloc_ct = rewloc_ct(ict);
F3trials_SR_ct = F3trials_SR_ct(ict);
F5trials_SR_ct = F5trials_SR_ct(ict);
total_SR_ct = total_SR_ct(ict);

% Reward zones: [67:86] [101:120] [135:154]
RZ1 = 67:86;
RZ2 = 101:120;
RZ3 = 135:154;

% First 3 trials w/ opto
F3zone1 = F3trials_SR_3opto(ismember(rewloc_3opto, RZ1));
F3zone2 = F3trials_SR_3opto(ismember(rewloc_3opto, RZ2));
F3zone3 = F3trials_SR_3opto(ismember(rewloc_3opto, RZ3));
allzone1 = total_SR_3opto(ismember(rewloc_3opto, RZ1));
allzone2 = total_SR_3opto(ismember(rewloc_3opto, RZ2));
allzone3 = total_SR_3opto(ismember(rewloc_3opto, RZ3));

% ctrl trials
F3zone1_c = F3trials_SR_ctrl(ismember(rewloc_ctrl, RZ1));
F3zone2_c = F3trials_SR_ctrl(ismember(rewloc_ctrl, RZ2));
F3zone3_c = F3trials_SR_ctrl(ismember(rewloc_ctrl, RZ3));

f1 = figure;
comparestatsplots(F3zone1,F3zone1_c,'b','r',[1 2]); hold on
comparestatsplots(F3zone2,F3zone2_c,'b','r',[4 5]);
comparestatsplots(F3zone3,F3zone3_c,'b','r',[7 8]);
xlim([0 9])
xticks([1 2 4 5 7 8])
xticklabels({'Opto: RZ1', 'Non-opto: RZ1', 'Opto: RZ2', 'Non-opto: RZ2', 'Opto: RZ3', 'Non-opto: RZ3'})
title({'Success rates of first 3 trials',' '})
ylabel('Success rate')
ylim([0 1])
hold off

allzone1_c = total_SR_ctrl(ismember(rewloc_ctrl, RZ1));
allzone2_c = total_SR_ctrl(ismember(rewloc_ctrl, RZ2));
allzone3_c = total_SR_ctrl(ismember(rewloc_ctrl, RZ3));
f2 = figure;
comparestatsplots(allzone1,allzone1_c,'b','r',[1 2]); hold on
comparestatsplots(allzone2,allzone2_c,'b','r',[4 5]);
comparestatsplots(allzone3,allzone3_c,'b','r',[7 8]);
xlim([0 9])
xticks([1 2 4 5 7 8])
xticklabels({'Opto: RZ1', 'Non-opto: RZ1', 'Opto: RZ2', 'Non-opto: RZ2', 'Opto: RZ3', 'Non-opto: RZ3'})
title({'First 3 trials w/ Opto vs. w/o Opto: Success rates of epochs',' '})
ylabel('Success rate')
ylim([0 1])
hold off

% First 5 trials w/ opto
F5zone1 = F5trials_SR_5opto(ismember(rewloc_5opto, RZ1));
F5zone2 = F5trials_SR_5opto(ismember(rewloc_5opto, RZ2));
F5zone3 = F5trials_SR_5opto(ismember(rewloc_5opto, RZ3));

allzone1 = total_SR_5opto(ismember(rewloc_5opto, RZ1));
allzone2 = total_SR_5opto(ismember(rewloc_5opto, RZ2));
allzone3 = total_SR_5opto(ismember(rewloc_5opto, RZ3));

% First 5 trials w/o opto
F5zone1_c = F5trials_SR_ctrl(ismember(rewloc_ctrl, RZ1));
F5zone2_c = F5trials_SR_ctrl(ismember(rewloc_ctrl, RZ2));
F5zone3_c = F5trials_SR_ctrl(ismember(rewloc_ctrl, RZ3));

f3 = figure;
comparestatsplots(F5zone1,F5zone1_c,'b','r',[1 2]); hold on
comparestatsplots(F5zone2,F5zone2_c,'b','r',[4 5]);
comparestatsplots(F5zone3,F5zone3_c,'b','r',[7 8]);
xlim([0 9])
xticks([1 2 4 5 7 8])
xticklabels({'Opto: RZ1', 'Non-opto: RZ1', 'Opto: RZ2', 'Non-opto: RZ2', 'Opto: RZ3', 'Non-opto: RZ3'})
title({'Success rates of first 5 trials',' '})
ylabel('Success rate')
ylim([0 1])
hold off

allzone1_c = total_SR_ctrl(ismember(rewloc_ctrl, RZ1));
allzone2_c = total_SR_ctrl(ismember(rewloc_ctrl, RZ2));
allzone3_c = total_SR_ctrl(ismember(rewloc_ctrl, RZ3));

f4 = figure;
comparestatsplots(allzone1,allzone1_c,'b','r',[1 2]); hold on
comparestatsplots(allzone2,allzone2_c,'b','r',[4 5]);
comparestatsplots(allzone3,allzone3_c,'b','r',[7 8]);
xlim([0 9])
xticks([1 2 4 5 7 8])
xticklabels({'Opto: RZ1', 'Non-opto: RZ1', 'Opto: RZ2', 'Non-opto: RZ2', 'Opto: RZ3', 'Non-opto: RZ3'})
title({'First 5 trials w/ Opto vs. w/o Opto: Success rates of epochs',' '})
ylabel('Success rate')
ylim([0 1])
hold off

% % Probes w/ opto
% F3zone1 = F3trials_SR_pbopto(ismember(rewloc_pbopto, RZ1));
% F3zone2 = F3trials_SR_pbopto(ismember(rewloc_pbopto, RZ2));
% F3zone3 = F3trials_SR_pbopto(ismember(rewloc_pbopto, RZ3));
% F5zone1 = F5trials_SR_pbopto(ismember(rewloc_pbopto, RZ1));
% F5zone2 = F5trials_SR_pbopto(ismember(rewloc_pbopto, RZ2));
% F5zone3 = F5trials_SR_pbopto(ismember(rewloc_pbopto, RZ3));
allzone1 = total_SR_pbopto(ismember(rewloc_pbopto, RZ1));
allzone2 = total_SR_pbopto(ismember(rewloc_pbopto, RZ2));
allzone3 = total_SR_pbopto(ismember(rewloc_pbopto, RZ3));
% 
% % Probes w/o opto
% F3zone1_c = F3trials_SR_ctrl(ismember(rewloc_ctrl, RZ1));
% F3zone2_c = F3trials_SR_ctrl(ismember(rewloc_ctrl, RZ2));
% F3zone3_c = F3trials_SR_ctrl(ismember(rewloc_ctrl, RZ3));
% [h1,p1,ci1,stats1] = ttest2(F3zone1,F3zone1_c);
% [h2,p2,ci2,stats2] = ttest2(F3zone2,F3zone2_c);
% [h3,p3,ci3,stats3] = ttest2(F3zone3,F3zone3_c);
% disp('Probe trials w opto: F3 SR vs. F5 SR vs. Total SR')
% disp('    RZ1       RZ2       RZ3')
% disp([p1 p2 p3])
% F5zone1_c = F5trials_SR_ctrl(ismember(rewloc_ctrl, RZ1));
% F5zone2_c = F5trials_SR_ctrl(ismember(rewloc_ctrl, RZ2));
% F5zone3_c = F5trials_SR_ctrl(ismember(rewloc_ctrl, RZ3));
% [h1,p1,ci1,stats1] = ttest2(F5zone1,F5zone1_c);
% [h2,p2,ci2,stats2] = ttest2(F5zone2,F5zone2_c);
% [h3,p3,ci3,stats3] = ttest2(F5zone3,F5zone3_c);
% disp([p1 p2 p3])
allzone1_c = total_SR_ctrl(ismember(rewloc_ctrl, RZ1));
allzone2_c = total_SR_ctrl(ismember(rewloc_ctrl, RZ2));
allzone3_c = total_SR_ctrl(ismember(rewloc_ctrl, RZ3));
f5 = figure;
comparestatsplots(allzone1,allzone1_c,'b','r',[1 2]); hold on
comparestatsplots(allzone2,allzone2_c,'b','r',[4 5]);
comparestatsplots(allzone3,allzone3_c,'b','r',[7 8]);
xlim([0 9])
xticks([1 2 4 5 7 8])
xticklabels({'Opto: RZ1', 'Non-opto: RZ1', 'Opto: RZ2', 'Non-opto: RZ2', 'Opto: RZ3', 'Non-opto: RZ3'})
title({'Probes w/ Opto vs. w/o Opto: Success rates of epochs',' '})
ylabel('Success rate')
ylim([0 1])
hold off
% [ah1,ap1,aci1,astats1] = ttest2(allzone1,allzone1_c);
% [ah2,ap2,aci2,astats2] = ttest2(allzone2,allzone2_c);
% [ah3,ap3,aci3,astats3] = ttest2(allzone3,allzone3_c);
% disp([ap1 ap2 ap3])
% 
% [h11,p11,ci11,stats11] = ttest2(F3zone1,F3zone1_ct);
% [h22,p22,ci22,stats22] = ttest2(F3zone2,F3zone2_ct);
% [h33,p33,ci33,stats33] = ttest2(F3zone3,F3zone3_ct);
% disp('Probe trials w opto vs. non-opto days: F3 SR vs. F5 SR vs. Total SR')
% disp('    RZ1       RZ2       RZ3')
% disp([p11 p22 p33])
% [h11,p11,ci11,stats11] = ttest2(F5zone1,F5zone1_ct);
% [h22,p22,ci22,stats22] = ttest2(F5zone2,F5zone2_ct);
% [h33,p33,ci33,stats33] = ttest2(F5zone3,F5zone3_ct);
% disp([p11 p22 p33])
% [ah11,ap11,aci11,astats11] = ttest2(allzone1,allzone1_ct);
% [ah22,ap22,aci22,astats22] = ttest2(allzone2,allzone2_ct);
% [ah33,ap33,aci33,astats33] = ttest2(allzone3,allzone3_ct);
% disp([ap11 ap22 ap33])
saveas(f1,'H:\E186\E186\success_rate\F3_1.png','png')
saveas(f2,'H:\E186\E186\success_rate\F3_2.png','png')
saveas(f3,'H:\E186\E186\success_rate\F5_1.png','png')
saveas(f4,'H:\E186\E186\success_rate\F5_2.png','png')
saveas(f5,'H:\E186\E186\success_rate\PB_1.png','png')

disp('Done!')