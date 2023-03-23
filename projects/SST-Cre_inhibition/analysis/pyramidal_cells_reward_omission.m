% Zahra
clear all; close all
grayColor = [.7 .7 .7];

for day=[19,25]
    fl=dir(fullfile(sprintf('Y:\\sstcre_imaging\\e200\\%i',day), '**\*Fall.mat'));
    load(fullfile(fl.folder,fl.name));
    
    dff = redo_dFF(F, 31.25, 20, Fneu);
    
    range=3;
    bin=0.2;
    rewardsonly=rewards>=1;
    cs=rewards==0.5;
    % runs for all cells
    [binnedPerireward,allbins,rewdFF] = perirewardbinnedactivity(dff',rewardsonly,timedFF,range,bin); %rewardsonly if mapping to reward
    % find av velocity too    
    idx = find(rewardsonly);
    periCSvel = zeros(length(idx),length(allbins));
    for iid=1:length(idx)
        rn = (idx(iid)-(range/0.2):idx(iid)+(range/0.2)-1);
        if max(rn)>40000
            rn(find(rn>40000))=NaN;
        end
        periCSvel(iid,:)=forwardvel(rn);
    end     
     periCSveld_av = nanmean(periCSvel,1);
        
    
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
    subplot(2,1,1);
    A = normalize(binnedPerireward,2);
    maxA = max(A, [], 2);
    [~, index] = sort(maxA);
    B    = A(index, :);
    imagesc(B); colorbar;
    ticks = [1:5:range*10, range*10];
    xticks(ticks)
    xline(median(ticks),'-k','R'); %{'Conditioned', 'stimulus'}
    xline(median(ticks)-2.5, '-.r','CS'); %{'Conditioned', 'stimulus'}
    xticklabels([allbins(1:5:end) range]);
    xlabel('seconds')
    ylabel('dF/F') 
    subplot(2,1,2);
    plot(periCSveld_av, 'Color', grayColor)
    xline(median(ticks),'-k','R'); %{'Conditioned', 'stimulus'}
    xline(median(ticks)-2.5, '-.r','CS'); %{'Conditioned', 'stimulus'}
    xlabel('seconds')
    ylabel('average velocity') 
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

