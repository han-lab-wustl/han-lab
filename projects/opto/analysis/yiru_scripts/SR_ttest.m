% probe days: 24 25 27 28 29 31 32 control days: 21 22 26 30 34
% 2 probe trials + 3 ep trials: 35 36 37 39 40 41 44 45 46, control: 34 38 43
% 5 ep trials: 48 49 51 53 54 55 57 58, control: 52 56 59 60

% group them (ignore stim types) and run ttest2
probes_day = [24 25 27 28 29 31 32];
pbs2_eps3_day = [35 36 37 39 40 41 44 45 46];
eps5_day = [48 49 51 53 54 55 57 58];
ctrl_day = [21 22 26 30 34 38 43 52 56 59 60];
all_opto = [probes_day pbs2_eps3_day eps5_day];

% Reward zones: [67:86] [101:120] [135:154]
RZ1 = 67:86;
RZ2 = 101:120;
RZ3 = 135:154;

% create a matrix with RZ#, ep#, SR(, Dxx/opto treatment)
ctrl_ep2 = zeros(length(ctrl_day),4);
ctrl_ep3 = zeros(length(ctrl_day),4);
opto_ep = zeros(length(all_opto),4);

load('H:\E186\E186\alldays_info.mat')
for c = 1:length(ctrl_day)
    if length([epoch.epoch]) < 3
        continue
    end
    ctrl_ep2(c,4) = ctrl_day(c);
    day = ['D' num2str(ctrl_day(c))];
    idx = find([data.day]== day);
    epoch = data(idx).day_eps;
    rl2 = epoch(2).RewLoc;
    ctrl_ep2(c,2) = 2;
    if ismember(rl2, RZ1)
        ctrl_ep2(c,1) = 1
    elseif ismember(rl2, RZ2)
        ctrl_ep2(c,1) = 2
    elseif ismember(rl2, RZ3)
        ctrl_ep2(c,1) = 3
    end
    ctrl_ep2(c,3) = sum(epoch(2).success_info)/length(epoch(2).success_info)
    if length([epoch.epoch]) < 4
        continue
    end
    ctrl_ep3(c,4) = ctrl_day(c);
   
    rl3 = epoch(3).RewLoc;
    ctrl_ep3(c,2) = 3;
    if ismember(rl3, RZ1)
        ctrl_ep3(c,1) = 1
    elseif ismember(rl3, RZ2)
        ctrl_ep3(c,1) = 2
    elseif ismember(rl3, RZ3)
        ctrl_ep3(c,1) = 3
    end
    ctrl_ep3(c,3) = sum(epoch(3).success_info)/length(epoch(3).success_info)
end
ctrl_ep2 = ctrl_ep2(ctrl_ep2(:,2)>0,:);
ctrl_ep3 = ctrl_ep3(ctrl_ep3(:,2)>0,:);

for o = 1:length(all_opto)
    day = ['D' num2str(all_opto(o))];
    idx = find([data.day]== day);
    epoch = data(idx).day_eps;
    for e = 2:3
        if e==2 && length([epoch.epoch])<3
            continue
        end
        if e==3 && length([epoch.epoch])<4
            continue
        end
        if sum(epoch(e).opto_stim) >0 || sum(epoch(e).probe_opto)>0
            rl = epoch(e).RewLoc;
            opto_ep(o,2) = e;
            if ismember(rl, RZ1)
                opto_ep(o,1) = 1
            elseif ismember(rl, RZ2)
                opto_ep(o,1) = 2
            elseif ismember(rl, RZ3)
                opto_ep(o,1) = 3
            end
            opto_ep(o,3) = sum(epoch(e).success_info)/length(epoch(e).success_info)
            if sum(epoch(e).opto_stim) == 0 && sum(epoch(e).probe_opto) == 2
                opto_ep(o,4) = 1
            elseif sum(epoch(e).opto_stim) == 3 && sum(epoch(e).probe_opto) == 2
                opto_ep(o,4) = 2
            elseif sum(epoch(e).opto_stim) == 5 && sum(epoch(e).probe_opto) == 0
                opto_ep(o,4) = 3
            end
        end
    end
end
opto_ep = opto_ep(opto_ep(:,2)>0,:);

ep2 = find(opto_ep(:,2)==2)
opto_ep2 = opto_ep(ep2,:)
rz1 = find(opto_ep2(:,1)==1)
oep2_rz1 = opto_ep2(rz1,:)
rz2 = find(opto_ep2(:,1)==2)
oep2_rz2 = opto_ep2(rz2,:)
rz3 = find(opto_ep2(:,1)==3)
oep2_rz3 = opto_ep2(rz3,:)

ep3 = find(opto_ep(:,2)==3)
opto_ep3 = opto_ep(ep3,:)
rz1 = find(opto_ep3(:,1)==1)
oep3_rz1 = opto_ep3(rz1,:)
rz2 = find(opto_ep3(:,1)==2)
oep3_rz2 = opto_ep3(rz2,:)
rz3 = find(opto_ep3(:,1)==3)
oep3_rz3 = opto_ep3(rz3,:)

rz1 = find(ctrl_ep2(:,1)==1)
cep2_rz1 = ctrl_ep2(rz1,:)
rz2 = find(ctrl_ep2(:,1)==2)
cep2_rz2 = ctrl_ep2(rz2,:)
rz3 = find(ctrl_ep2(:,1)==3)
cep2_rz3 = ctrl_ep2(rz3,:)

rz1 = find(ctrl_ep3(:,1)==1)
cep3_rz1 = ctrl_ep3(rz1,:)
rz2 = find(ctrl_ep3(:,1)==2)
cep3_rz2 = ctrl_ep3(rz2,:)
rz3 = find(ctrl_ep3(:,1)==3)
cep3_rz3 = ctrl_ep3(rz3,:)

f1=figure('Renderer','painters','Position', [20 20 1000 700]);
comparestatsplots(oep2_rz1(:,3), cep2_rz1(:,3),'b','r',[1 2]); hold on
comparestatsplots(oep2_rz2(:,3), cep2_rz2(:,3),'b','r',[4 5])
comparestatsplots(oep2_rz3(:,3), cep2_rz3(:,3),'b','r',[7 8])
xlim([0 9])
xticks([1 2 4 5 7 8])
ylabel('Success Rate')
ylim([0 1])
xticklabels({'Opto: RZ1', 'Non-opto: RZ1', 'Opto: RZ2', 'Non-opto: RZ2', 'Opto: RZ3', 'Non-opto: RZ3'})
title({'Opto days vs. Non-opto days: Epoch 2',' '})
hold off

f2=figure('Renderer','painters','Position', [20 20 1000 700]);
comparestatsplots(oep3_rz1(:,3), cep3_rz1(:,3),'b','r',[1 2]); hold on
comparestatsplots(oep3_rz2(:,3), cep3_rz2(:,3),'b','r',[4 5])
comparestatsplots(oep3_rz3(:,3), cep3_rz3(:,3),'b','r',[7 8])
xlim([0 9])
xticks([1 2 4 5 7 8])
ylim([0 1])
ylabel('Success Rate')
xticklabels({'Opto: RZ1', 'Non-opto: RZ1', 'Opto: RZ2', 'Non-opto: RZ2', 'Opto: RZ3', 'Non-opto: RZ3'})
title({'Opto days vs. Non-opto days: Epoch 3',' '})
hold off

f3=figure('Renderer','painters','Position', [20 20 1000 700]);
comparestatsplots(opto_ep2(:,3), ctrl_ep2(:,3),'b','r',[1 2]); hold on
scatter(0.60,opto_ep2(opto_ep2(:,4)==1,3),30,'MarkerEdgeAlpha',0.6, 'LineWidth', 1.5)
scatter(0.70,opto_ep2(opto_ep2(:,4)==2,3),30,'MarkerEdgeAlpha',0.6, 'LineWidth', 1.5)
scatter(0.85,opto_ep2(opto_ep2(:,4)==3,3),30,'MarkerEdgeAlpha',0.6, 'LineWidth', 1.5)
comparestatsplots(opto_ep3(:,3), ctrl_ep3(:,3),'b','r',[4 5])
scatter(3.60,opto_ep3(opto_ep3(:,4)==1,3),30,'MarkerEdgeAlpha',0.6, 'LineWidth', 1.5)
scatter(3.70,opto_ep3(opto_ep3(:,4)==2,3),30,'MarkerEdgeAlpha',0.6, 'LineWidth', 1.5)
scatter(3.85,opto_ep3(opto_ep3(:,4)==3,3),30,'MarkerEdgeAlpha',0.6, 'LineWidth', 1.5)
comparestatsplots(opto_ep(:,3), [ctrl_ep2(:,3); ctrl_ep3(:,3)],'b','r',[7 8])
scatter(6.60,opto_ep(opto_ep(:,4)==1,3),30,'MarkerEdgeAlpha',0.6, 'LineWidth', 1.5)
scatter(6.70,opto_ep(opto_ep(:,4)==2,3),30,'MarkerEdgeAlpha',0.6, 'LineWidth', 1.5)
scatter(6.85,opto_ep(opto_ep(:,4)==3,3),30,'MarkerEdgeAlpha',0.6, 'LineWidth', 1.5)
xlim([0 9])
xticks([1 2 4 5 7 8])
ylim([0 1])
ylabel('Success Rate')
xticklabels({'Opto: ep2', 'Non-opto: ep2', 'Opto: ep3', 'Non-opto: ep3', 'Opto: all eps', 'Non-opto: all eps'})
title({'Epochs in Opto days vs. Non-opto days',' '})
hold off

lb = string({'Opto:ep2','Non-opto:ep2','Opto:ep2','Non-opto:ep2','Opto:ep2','Non-opto:ep2','Opto:ep2','Non-opto:ep2'})
figure('Renderer','painters','Position', [20 20 1000 700]);
t1 = tiledlayout('flow')
sr_comparisonplots(opto_ep2, ctrl_ep2, 'b', 'r', lb)

lb = string({'Opto:ep3','Non-opto:ep3','Opto:ep3','Non-opto:ep3','Opto:ep3','Non-opto:ep3','Opto:ep3','Non-opto:ep3'})
figure('Renderer','painters','Position', [20 20 1000 700]);
t2 = tiledlayout('flow')
sr_comparisonplots(opto_ep3, ctrl_ep3, 'b', 'r', lb)

lb = string({'Opto:all eps','Non-opto:all eps','Opto:all eps','Non-opto:all eps','Opto:all eps','Non-opto:all eps','Opto:all eps','Non-opto:all eps'})
figure('Renderer','painters','Position', [20 20 1000 700]);
t3 = tiledlayout('flow')
sr_comparisonplots(opto_ep, [ctrl_ep2; ctrl_ep3], 'b', 'r', lb)

lb = string({'Opto:ep2','Non-opto:ep2','Opto:ep2','Non-opto:ep2','Opto:ep2','Non-opto:ep2','Opto:ep2','Non-opto:ep2'})
figure('Renderer','painters','Position', [20 20 1000 700]);
t4 = tiledlayout('flow')
sr_comparisonplots_recentdays(opto_ep2, ctrl_ep2, 'b', 'r', lb)

lb = string({'Opto:ep3','Non-opto:ep3','Opto:ep3','Non-opto:ep3','Opto:ep3','Non-opto:ep3','Opto:ep3','Non-opto:ep3'})
figure('Renderer','painters','Position', [20 20 1000 700]);
t5 = tiledlayout('flow')
sr_comparisonplots_recentdays(opto_ep3, ctrl_ep3, 'b', 'r', lb)

lb = string({'Opto:all eps','Non-opto:all eps','Opto:all eps','Non-opto:all eps','Opto:all eps','Non-opto:all eps','Opto:all eps','Non-opto:all eps'})
figure('Renderer','painters','Position', [20 20 1000 700]);
t6 = tiledlayout('flow')
sr_comparisonplots_recentdays(opto_ep, [ctrl_ep2; ctrl_ep3], 'b', 'r', lb)

saveas(f1,'H:\E186\E186\success_rate\ep2_day_comparison.png','png')
saveas(f2,'H:\E186\E186\success_rate\ep3_day_comparison.png','png')
saveas(f3,'H:\E186\E186\success_rate\eps_day_comparison.png','png')

saveas(t1,'H:\E186\E186\success_rate\ep2_opto_comparison.png','png')
saveas(t2,'H:\E186\E186\success_rate\ep3_opto_comparison.png','png')
saveas(t3,'H:\E186\E186\success_rate\eps_opto_comparison.png','png')
saveas(t4,'H:\E186\E186\success_rate\ep2_recentdays_comparison.png','png')
saveas(t5,'H:\E186\E186\success_rate\ep3_recentdays_comparison.png','png')
saveas(t6,'H:\E186\E186\success_rate\eps_recentdays_comparison.png','png')