% Zahra
% day to day analysis for rewards in the dark using suite2p ROIS
% SST cre experiment
% want to plot and see if there is any activity of cells during CS/US
clear all

load('Y:\sstcre_imaging\e200\32\230329_ZD_000_000\suite2p\plane0\Fall.mat')
%%
% dff = redo_dFF(F, 31.25, 20, Fneu);
% save('Y:\sstcre_imaging\e200\20\230315_ZD_000_000\suite2p\plane0\Fall.mat', 'dFF', '-append')
dFF = dFF';
dFF = dFF(logical(iscell(:,1)),:);
range=10;
bin=0.1;
rewardsonly=rewards>=1;
cs=rewards==0.5;
% F = F(logical(iscell(:,1)),:);
% runs for all cells
[binnedPerireward,allbins,rewdFF] = perirewardbinnedactivity(dFF',cs,timedFF,range,bin); %rewardsonly if mapping to reward
[vbinnedPerireward,vallbins,vrewdFF] = perirewardbinnedactivity(forwardvel',cs,timedFF,range,bin); %rewardsonly if mapping to reward
[lbinnedPerireward,~,~] = perirewardbinnedactivity(licks',cs,timedFF,range,bin); %rewardsonly if mapping to reward
figure; imagesc(normalize(binnedPerireward,2)); hold on;
xline(100, 'w--', 'CS', 'LineWidth',3)
yyaxis right
plot(vbinnedPerireward, 'k', 'LineWidth',2); hold on
plot(lbinnedPerireward*50, 'ro')

%%
figure; plot(mean(binnedPerireward,1),'LineWidth',4); hold on;
xline(100, 'g--', 'CS', 'LineWidth',3)
yyaxis right
plot(vbinnedPerireward, 'k', 'LineWidth',2); hold on
plot(lbinnedPerireward*100, 'ro')
%%
for i=1:30%size(binnedPerireward,1)
    figure; plot(binnedPerireward(i,:),'LineWidth',4); hold on;
    xline(100, 'g--', 'CS', 'LineWidth',3)
    yyaxis right
    plot(vbinnedPerireward, 'k', 'LineWidth',2); hold on
    plot(lbinnedPerireward*100, 'ro')
end
%%
% plot all cells aligned to rewards
figure;
grayColor = [.7 .7 .7];    

for cellno=160:200 % plot each cell
    plot(binnedPerireward(cellno,:), 'Color', grayColor) %important distinction
    hold on;        
    % plot reward location as line
    xticks([1:5:200, 200])
    x1=xline(median([1:5:200, 200]),'-.b','CS'); %{'Conditioned', 'stimulus'}
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
    plot(spks(cellno,:),'k') % 2 in the first position is cell no
    hold on;
    set(gca,'XTick',[], 'YTick', [])
    set(gca,'visible','off')
end
% linkaxes([axs{:}],'xy')
copygraphics(gcf, 'BackgroundColor', 'none');
title(sprintf('Cell no. %03d', cellno));

%% 
% plot cell traces with behavior

for cellno=1:20
    figure;
    subplot(2,1,1)
    sp = F(cellno,:);
    plot(sp,'g') % 2 in the first position is cell no    
    hold on;
    plot(find(rewards==0.5), sp(rewards==0.5), 'bo', 'MarkerSize',7)
    plot(find(licks), sp(licks), 'ro', 'MarkerSize',3)
    subplot(2,1,2)    
    plot(forwardvel, 'k'); hold on
    plot(find(rewards==0.5), forwardvel(rewards==0.5), 'bo', 'MarkerSize',7)
    plot(find(licks), forwardvel(licks), 'ro', 'MarkerSize',3)  
end
% linkaxes([axs{:}],'xy')
copygraphics(gcf, 'BackgroundColor', 'none');
title(sprintf('Cell no. %03d', cellno));

