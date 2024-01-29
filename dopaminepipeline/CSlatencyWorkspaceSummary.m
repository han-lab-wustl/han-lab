clear all
close all
wrk_dir = uipickfiles('Prompt','Pick the Workspace you would like to add');
%%
figure;
hold on
for m = 1:length(wrk_dir)
    subplot(1,2,1)
    hold on
load(wrk_dir{m})
errorbar(1:length(CS_alldays_lick_gap),cellfun(@nanmean,CS_alldays_lick_gap),cellfun(@nanstd,CS_alldays_lick_gap)./sqrt(cellfun(@length,CS_alldays_lick_gap)))
hold on
xlabel('Days')
ylabel('Latency - S')
title('Lick Latency')
subplot(1,2,2)
hold on
errorbar(1:length(CS_alldays_stop_gap),cellfun(@nanmean,CS_alldays_stop_gap),cellfun(@nanstd,CS_alldays_stop_gap)./sqrt(cellfun(@length,CS_alldays_stop_gap)))
hold on
xlabel('Days')
ylabel('Latency - S')
title('Stop Latency')
end