% Zahra
% plot behavior (ROE, licks) of animals in SST experiment
% align to CS to see if the animal learns anything
% E201 (speedy) :|

% day 1 of new conditioning
% no rewards no VR
% letting the animal run for a while and manually delivering CS/US with
% numpad / in the paradigm

clear all
load('Y:\sstcre_troubleshooting\E201_14_Mar_2023_time(08_30_32).mat')
grayColor = [.7 .7 .7];

forwardvel=-0.013*VR.ROE(100:end)./diff(VR.time(99:end)); % skipping first 100 frames just to even out arrays
sol = VR.reward(100:end)==0.5; % codes for solenoid
reward = VR.reward(100:end)>0.5; 

figure;
plot(sol*50, 'g', 'LineWidth',3); hold on;
plot(reward*50, 'b', 'LineWidth',3); hold on;
test = VR.lickVoltage*1000;
plot(test,'r'); hold on;
plot(forwardvel, 'Color', grayColor)
% xlim([0 110000])

% peri cs velocity
range=30;
idx = find(sol);
periCSlick = zeros(length(idx),61);
periCSvel = zeros(length(idx),61);
for iid=1:length(idx)
    figure;
    rn = (idx(iid)-range:idx(iid)+range);    
%     periCSlick(iid,:)=VR.lickVoltage(rn);
    periCSvel(iid,:)=forwardvel(rn);
%     plot(periCSlick(iid,:), 'Color', grayColor); hold on;          
    plot(periCSvel(iid,:), 'Color', grayColor); hold on;
    x1=xline(31,'-.b',{'Conditioned', 'stimulus'});
end
% plot(nanmean(periCSlick, 1), 'r'); hold on;
plot(nanmean(periCSvel,1), 'k')

% plot CS triggered velocity changes
figure;
plot(periCSveld{d}', 'Color', grayColor); hold on;          
plot(periCSveld_av{d}, 'b');

xlim([0 61])
x1=xline(31,'-.b',{'Conditioned', 'stimulus'}); %{'Conditioned', 'stimulus'}, 'Reward'
title(sprintf("%s, day %s", animal , daynms{d}))
xlabel('frames')
ylabel('lick voltage')

