% Zahra
% day to day analysis for rewards in the dark using suite2p ROIS
% SST cre experiment
% want to plot and see if there is any activity of cells during CS/US
clear all
load('Z:\sstcre_imaging\e200\19\230314_ZD_000_000\suite2p\plane0\Fall.mat')

dff = redo_dFF(F, 31.25, 20, Fneu);

range=5;
bin=0.2;
rewardsonly=rewards>=1;
cs=rewards==0.5;
% runs for all cells
[binnedPerireward,allbins,rewdFF] = perirewardbinnedactivity(dff',rewardsonly,timedFF,range,bin); %rewardsonly if mapping to reward
    
%%
% plot all cells aligned to rewards
figure;
grayColor = [.7 .7 .7];    

for cellno=1:size(F,1) % plot each cell
    plot(binnedPerireward(cellno,:), 'Color', grayColor) %important distinction
    hold on;        
    % plot reward location as line
    xticks([1:5:50, 50])
    x1=xline(median([1:5:50, 50]),'-.b','Reward'); %{'Conditioned', 'stimulus'}
    xticklabels([allbins(1:5:end) range]);
    xlabel('seconds')
    ylabel('dF/F')        
end

%%
% plot all cell traces

fig=figure;
cells2plot=60; %size(F,1);
for cellno=40:cells2plot
    ax1=subplot(20,1,cellno-39);
    plot(dff(cellno,:),'k') % 2 in the first position is cell no
    hold on;
    set(gca,'XTick',[], 'YTick', [])
    set(gca,'visible','off')
end
% linkaxes([axs{:}],'xy')
copygraphics(gcf, 'BackgroundColor', 'none');
title(sprintf('Cell no. %03d', cellno));
