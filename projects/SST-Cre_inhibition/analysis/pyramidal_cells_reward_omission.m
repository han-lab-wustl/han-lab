% Zahra
clear all; close all
for day=8:21
    fl=dir(fullfile(sprintf('Z:\\sstcre_imaging\\e200\\%i',day), '**\*Fall.mat'));
    load(fullfile(fl.folder,fl.name));
    
    dff = redo_dFF(F, 31.25, 20, Fneu);
    
    range=5;
    bin=0.2;
    rewardsonly=rewards>=1;
    cs=rewards==0.5;
    % runs for all cells
    [binnedPerireward,allbins,rewdFF] = perirewardbinnedactivity(dff',rewardsonly,timedFF,range,bin); %rewardsonly if mapping to reward
        
    
    % plot all cells aligned to rewards
    % grayColor = [.7 .7 .7];    
    % 
    % figure;
    % for cellno=1:size(F,1) % plot each cell    
    %     plot(binnedPerireward(cellno,:), 'Color', grayColor) 
    %     hold on;        
    %     % plot reward location as line
    %     xticks([1:5:50, 50])
    %     x1=xline(median(1:50),'-.b','Reward'); %{'Conditioned', 'stimulus'}
    %     xticklabels([allbins(1:5:end) range]);
    %     xlabel('seconds')
    %     ylabel('dF/F')        
    % end
    
    figure;
    A = normalize(binnedPerireward,2);
    maxA = max(A, [], 2);
    [~, index] = sort(maxA);
    B    = A(index, :);
    imagesc(B); hold on;
    xticks([1:5:50, 50])
    xline(median([1:5:50, 50]),'-k','R'); %{'Conditioned', 'stimulus'}
    xline(median([1:5:50, 50])-2.5, '-.r','CS'); %{'Conditioned', 'stimulus'}
    xticklabels([allbins(1:5:end) range]);
    xlabel('seconds')
    ylabel('dF/F') 
    title(sprintf("E200, RR, Day %i of recording", day))
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

