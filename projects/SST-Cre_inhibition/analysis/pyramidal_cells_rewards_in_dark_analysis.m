% Zahra
% day to day analysis for rewards in the dark using suite2p ROIS
% SST cre experiment
% want to plot and see if there is any activity of cells during CS/US
clear all
days = [8:27];
mouse_name = "e200";
cnt = 1; %counter for days
for dy=days
    src = "Y:\sstcre_imaging";
    fmatfl = dir(fullfile(src, mouse_name, sprintf('%i',dy), '**\Fall.mat'));
    load(fullfile(fmatfl.folder, fmatfl.name))
    %%
    % dff = redo_dFF(F, 31.25, 20, Fneu);
    % save('Y:\sstcre_imaging\e200\20\230315_ZD_000_000\suite2p\plane0\Fall.mat', 'dFF', '-append')
    dFF = dFF';
    skews = cell2mat(cellfun(@(x) x.skew, stat, 'UniformOutput', false));
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
    perirew{cnt} = binnedPerireward; % individual cells
    vperirew{cnt} = vbinnedPerireward;
    lperirew{cnt} = lbinnedPerireward;    
    cnt = cnt+1;
end
%%
%plot summary figure
vmat = reshape(cell2mat(vperirew), [200,length(vperirew)]);
lmat = reshape(cell2mat(lperirew), [200,length(lperirew)]);
fmat = reshape(cell2mat(cellfun(@(x) mean(x), perirew, 'UniformOutput',false)), [200,length(lperirew)]);
ax1 = subplot(3,1,1);
imagesc(vmat')
xticks([0:20:200])
xticklabels([-10:2:10])
colormap(ax1,gray)
title('binned velocity')
ylabel('days')
ax2 = subplot(3,1,2);
imagesc(lmat')
xticks([0:20:200])
xticklabels([-10:2:10])
colormap(ax2,autumn)
title('binned licks')
ylabel('days')
ax3 = subplot(3,1,3);
imagesc(normalize(fmat',2))
xticks([0:20:200])
xticklabels([-10:2:10])
colormap(ax3,jet)
title('binned std dff')
ylabel('days')
sgtitle('cs triggered averages')

ax4 = subplot(2,1,1);
imagesc(vmat')
xticks([0:20:200])
xticklabels([-10:2:10])
colormap(ax4,gray)
title('binned velocity')
ylabel('days')
ax5 = subplot(2,1,2);
imagesc(normalize(fmat',2))
xline(100, 'k--', 'CS', 'LineWidth',3)
xticks([0:20:200])
xticklabels([-10:2:10])
colormap(ax5,jet)
title('binned average dff')
ylabel('days')

%%
% per cell
% pick a random cell
c = randi([30 50],1,1);
vmat = reshape(cell2mat(vperirew), [200,length(vperirew)]);
lmat = reshape(cell2mat(lperirew), [200,length(lperirew)]);
fmat = reshape(cell2mat(cellfun(@(x) x(c,:), perirew, 'UniformOutput',false)), [200,length(lperirew)]);
ax1 = subplot(3,1,1);
imagesc(vmat')
xticks([0:20:200])
xticklabels([-10:2:10])
colormap(ax1,gray)
title('binned velocity')
ylabel('days')
ax2 = subplot(3,1,2);
imagesc(lmat')
xticks([0:20:200])
xticklabels([-10:2:10])
colormap(ax2,autumn)
title('binned licks')
ylabel('days')
ax3 = subplot(3,1,3);
imagesc(normalize(fmat',2))
xticks([0:20:200])
xticklabels([-10:2:10])
colormap(ax3,jet)
title('binned average dff')
ylabel('days')
sgtitle('cs triggered averages')
%%
% find cells corr with velocity
corrs = {};
for day=1:length(perirew)
    f = perirew{day};
    corr_ = {}; 
    for c=1:size(f,1)
        fc = f(c,:);
        vc = vperirew{day};
        corr = corrcoef(fc,vc);
        corr_{c} = corr(2,1); % take diagonal values
    end
    corrs{day} = corr_;    
end
%%
% early days vs late days or all days
avcorr = {};
for day=1:length(perirew)
    poscorr = cell2mat(corrs{day})>0.7;
    negcorr = cell2mat(corrs{day})<-0.7;
    notcorr = (cell2mat(corrs{day})>-0.7 & cell2mat(corrs{day})<0.7);
    ccorr = perirew{day}(notcorr,:);
    cpos = perirew{day}(poscorr,:);
    cneg = perirew{day}(negcorr,:);
%     figure;
%     ax1 = subplot(2,1,1);
%     imagesc(vmat(:,day)')
%     xticks([0:20:200])
%     xticklabels([-10:2:10])
%     colormap(ax1,gray)
%     title('binned velocity')
%     ylabel('days')
%     ax2 = subplot(2,1,2);
%     imagesc(normalize(ccorr,2))
%     colormap(ax2,jet)
%     title('binned dff')
%     ylabel('cells')
%     sgtitle(sprintf('day %i of RR', day))
    avcorr{day} = mean(ccorr, 1); % average of uncorrelated cells
end

avcorr = reshape(cell2mat(avcorr), [200,length(avcorr)]);

% plot average of non correlated cells
ax1 = subplot(3,1,1);
imagesc(vmat')
xticks([0:20:200])
xticklabels([-10:2:10])
colormap(ax1,gray)
title('binned velocity')
ylabel('days')
ax2 = subplot(3,1,2);
imagesc(lmat')
xticks([0:20:200])
xticklabels([-10:2:10])
colormap(ax2,autumn)
title('binned licks')
ylabel('days')
ax3 = subplot(3,1,3);
imagesc(normalize(avcorr',2))
xticks([0:20:200])
xticklabels([-10:2:10])
colormap(ax3,jet)
title('binned average dff')
ylabel('days')
sgtitle('cs triggered averages')
%%

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

