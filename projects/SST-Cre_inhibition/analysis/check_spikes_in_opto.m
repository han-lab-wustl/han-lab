% Zahra
% ep 2 days
% there is a bug in trialnum from VRalign/startend - ask Gerardo
% above bug is fixed in commit from 6/12/23
clear all
srcdir = 'Z:\sstcre_imaging';
animal = 'e201';
grayColor = [.7 .7 .7];
days = [55 58 61 64 67 70 73 77 80];% 83 86 88 91];%[55 58 61 64 67 70 73 77 80 83 86 88];%[65 68    71    74    78    81    84    87    90];

%[55 58 61 64 67 70 73 77 80]; % opto sequence, ep2 5 trials, ep3 5 trials, control
for d=1:length(days)
    clearvars -except srcdir animal grayColor days d
    fmatfl = dir(fullfile(srcdir, animal, string(days(d)), "**\Fall.mat")); 
    load(fullfile(fmatfl.folder,fmatfl.name));
    eps = find(changeRewLoc);
    % ep 2
    eprng = eps(2):eps(3);
    try  % to compare to
        eprng_comp = eps(3):eps(4);
    catch
        eprng_comp= eps(3):length(changeRewLoc);
    end
    % get xy axis
    stat_iscell = stat(logical(iscell(:,1)));
    if ~(size(stat_iscell,2)==size(all.dff,1)) % check if same size as all.dff (sometimes cells are not removed) 
        if exist('cell2remove', 'var') % check if cell2remove var exists
            stat_cell2remove = stat_iscell(~logical(cell2remove)&(~logical(remove_iscell)));
        else
            stat_cell2remove = stat_iscell((~logical(remove_iscell)));
        end
    else
        stat_cell2remove = stat_iscell;
    end
    Ypix = cellfun(@(x) x.ypix, stat_cell2remove, 'UniformOutput', false);
    topbordercells = zeros(1,length(Ypix)); % bool of cells at the top border
    for yy=1:length(Ypix) % idea is to remove these cells
        if sum(Ypix{yy}<100)>0
            topbordercells(yy)=1;
        end
    end
    % visualize
%     stat_topbordercells = stat_cell2remove(logical(topbordercells));
%     figure;
%     imagesc(ops.meanImg)
%     colormap('gray')
%     hold on;
%     for cell=1:length(stat_topbordercells)%length(commoncells)        
%         plot(stat_topbordercells{cell}.xpix, stat_topbordercells{cell}.ypix);         
%     end
    % only get cells > y pix of 100

    dff = all.dff(~logical(topbordercells),:);
    neural_data = dff(:,eprng);
    t = timedFF(eprng);  
    mask = (trialnum(eprng)>=3) & (trialnum(eprng)<8);
    opto = neural_data(:,mask);
    topto = t(mask);
    opto_comp = all.dff(:,eprng_comp);
    mask = (trialnum(eprng_comp)>=3) & (trialnum(eprng_comp)<8);
    opto_comp = opto_comp(:,mask);
    % take mean across time
    opto_mean = mean(opto, 2);
    opto_comp_mean = mean(opto_comp, 2);
    opto_means{d} = opto_mean;
    opto_comp_means{d} = opto_comp_mean;
%     figure; plot(1,opto_mean, 'ro'); hold on; plot(2, opto_comp_mean, 'ko')
%     for cell=1:length(opto_mean)
%         plot([1 2], [opto_mean(cell), opto_comp_mean(cell)], 'Color', grayColor); 
%     end
%     xlim([0 3]);
%     xticks([1 2])
%     ylabel('mean dff')
%     xticklabels([{'ep2, first 5 trials'}, {'ep3, first 5 trials'}])
%     title(sprintf('%s day%i, opto ep2', animal, days(d)))
%     % num spikes
%     numspikes_opto_ep2(d) = (sum(sum(opto>0, 2))/size(opto,1))/(max(topto)-min(topto));  % total spikes/number of cells/per s    
%     % makes raster plot of spks with ypos, licks, rew
%     pos = ybinned(eprng);
%     lks = licks(eprng);
%     rews = rewards(eprng);
%     pos_=pos(mask);
%     figure; subplot(2,1,1)
%     imagesc(normalize(opto,1));
%     xlim([0 size(opto,2)])
%     ylabel('spks')
%     subplot(2,1,2)
%     plot(pos_,'k'); hold on;
%     plot(find(lks(mask)),pos_(find(lks(mask))),'r.');
%     plot(find(rews(mask)==1),pos_(find(rews(mask)==1)), 'b.')
%     xlim([0 size(opto,2)])
%     sgtitle(sprintf('%s, day %i, opto ep2', animal, d))
end
figure; 
days_to_plot = length(opto_means); %length(opto_means)
for d=1:days_to_plot%length(opto_means)
    plot(1,opto_means{d}, 'ro'); hold on; plot(2, opto_comp_means{d}, 'ko')
    for cell=1:length(opto_means{d})
        plot([1 2], [opto_means{d}(cell), opto_comp_means{d}(cell)], 'Color', grayColor); 
    end
end
xlim([0 3]);
xticks([1 2])
ylabel('mean dff')
xticklabels([{'ep2, first 5 trials'}, {'ep3, first 5 trials'}])
title(sprintf('%s (%i days), opto ep2', animal, days_to_plot))
savefig('C:\Users\Han\Box\neuro_phd_stuff\han_2023\figures\e201_optoep2_meandff.fig')
% plot difference
figure;
for d=1:days_to_plot%length(opto_means)
    plot(1,diff([opto_means{d}; opto_comp_means{d}]), 'ro'); hold on;    
end
yline(mean(diff([cell2mat(cellfun(@(x) x', opto_means, 'UniformOutput', false)); ...
    cell2mat(cellfun(@(x) x', opto_comp_means, 'UniformOutput', false))]), 'omitnan'))
xlim([0 2]);
xticks([1])
ylabel('mean dff')
xticklabels(['ep2-ep3, first 5 trials'])
title(sprintf('%s (%i days), opto ep2', animal, days_to_plot))
savefig('C:\Users\Han\Box\neuro_phd_stuff\han_2023\figures\e201_optoep2_meandff_diff.fig')
%%
% ep 3
days = [56    59    62    65    68    71    75    78 82 84 87 89];%[66    69    72    75    79    82 88];

for d=1:length(days)
    fmatfl = dir(fullfile(srcdir, animal, string(days(d)), "**\Fall.mat")); 
    load(fullfile(fmatfl.folder,fmatfl.name));
    eps = find(changeRewLoc);
    % ep 2
    try
        eprng = eps(3):eps(4);
    catch
        eprng = eps(3):length(changeRewLoc);
    end
    eprng_comp = eps(2):eps(3);
    neural_data = all.dff(:,eprng);
    t = timedFF(eprng);    
    mask = (trialnum(eprng)>=3) & (trialnum(eprng)<8);
    opto = neural_data(:,mask);
    topto = t(mask);
    opto_comp = all.dff(:,eprng_comp);
    mask = (trialnum(eprng_comp)>=3) & (trialnum(eprng_comp)<8);
    opto_comp = opto_comp(:,mask);
    % take mean across time
    opto_mean = mean(opto, 2);
    opto_comp_mean = mean(opto_comp, 2);
    opto_means{d} = opto_mean;
    opto_comp_means{d} = opto_comp_mean;
%     figure; plot(1,opto_mean, 'ro'); hold on; plot(2, opto_comp_mean, 'ko')
%     for cell=1:length(opto_mean)
%         plot([1 2], [opto_mean(cell), opto_comp_mean(cell)], 'Color', grayColor); 
%     end
%     xlim([0 3]);
%     xticks([1 2])
%     ylabel('mean dff')
%     xticklabels([{'ep3, first 5 trials'}, {'ep2, first 5 trials'}])
%     title(sprintf('%s day%i, opto ep3', animal, days(d)))
    % num spikes
%     numspikes_opto_ep3(d) = (sum(sum(opto>0, 2))/size(opto,1))/(max(topto)-min(topto));  % total spikes/number of cells/per s
%     pos = ybinned(eprng);
%     lks = licks(eprng);
%     rews = rewards(eprng);
%     pos_=pos(mask);
%     figure; subplot(2,1,1)
%     imagesc(normalize(opto,1));
%     xlim([0 size(opto,2)])
%     ylabel('spks')
%     subplot(2,1,2)
%     plot(pos_,'k'); hold on;
%     plot(find(lks(mask)),pos_(find(lks(mask))),'r.');
%     plot(find(rews(mask)==1),pos_(find(rews(mask)==1)), 'b.')
%     xlim([0 size(opto,2)])
%     sgtitle(sprintf('%s, day %i, opto ep3', animal, d))
end
figure; 
days_to_plot = length(opto_means); %length(opto_means)
for d=1:days_to_plot%length(opto_means)
    plot(1,opto_means{d}, 'ro'); hold on; plot(2, opto_comp_means{d}, 'ko')
    for cell=1:length(opto_means{d})
        plot([1 2], [opto_means{d}(cell), opto_comp_means{d}(cell)], 'Color', grayColor); 
    end
end
xlim([0 3]);
xticks([1 2])
ylabel('mean dff')
xticklabels([{'ep3, first 5 trials'}, {'ep2, first 5 trials'}])
title(sprintf('%s (%i days), opto ep3', animal, days_to_plot))
savefig('C:\Users\Han\Box\neuro_phd_stuff\han_2023\figures\e201_optoep3_meandff.fig')
% plot difference
figure;
for d=1:days_to_plot%length(opto_means)
    plot(1,diff([opto_means{d}; opto_comp_means{d}]), 'ro'); hold on;    
end
yline(mean(diff([cell2mat(cellfun(@(x) x', opto_means, 'UniformOutput', false)); ...
    cell2mat(cellfun(@(x) x', opto_comp_means, 'UniformOutput', false))]), 'omitnan'))
xlim([0 2]);
xticks([1])
ylabel('mean dff')
xticklabels(['ep3-ep2, first 5 trials'])
title(sprintf('%s (%i days), opto ep3', animal, days_to_plot))
savefig('C:\Users\Han\Box\neuro_phd_stuff\han_2023\figures\e201_optoep3_meandff_diff.fig')
%%
% compare to control ep 2 and 3
days =  [57    60    63    66    69    72    76    79 85];%[67    70    73    76    80    83    86    89]; %[57    60    63    66    69    72    76    79];
for d=1:length(days)
    fmatfl = dir(fullfile(srcdir, animal, string(days(d)), "**\Fall.mat")); 
    load(fullfile(fmatfl.folder,fmatfl.name))
    eps = find(changeRewLoc);
    % ep 2
    try
        eprng = eps(3):eps(4);
    catch
        eprng = eps(3):length(changeRewLoc);
    end
    eprng_comp = eps(2):eps(3);
    neural_data = all.dff(:,eprng);
    t = timedFF(eprng);    
    mask = (trialnum(eprng)>=3) & (trialnum(eprng)<8);
    opto = neural_data(:,mask);
    topto = t(mask);
    opto_comp = all.dff(:,eprng_comp);
    mask = (trialnum(eprng_comp)>=3) & (trialnum(eprng_comp)<8);
    opto_comp = opto_comp(:,mask);
    mask = (trialnum(eprng_comp)>=3) & (trialnum(eprng_comp)<8);
    opto_comp = opto_comp(:,mask);
    % take mean across time
    opto_mean = mean(opto, 2);
    opto_comp_mean = mean(opto_comp, 2);
    figure; plot(1,opto_mean, 'bo'); hold on; plot(2, opto_comp_mean, 'ko')
    for cell=1:length(opto_mean)
        plot([1 2], [opto_mean(cell), opto_comp_mean(cell)], 'Color', grayColor); 
    end
    xlim([0 3]);
    xticks([1 2])
    ylabel('mean dff')
    xticklabels([{'ep3, first 5 trials'}, {'ep2, first 5 trials'}])
    title(sprintf('%s day%i, control', animal, days(d)))
end
% figure;
% bar([mean(numspikes_opto_ep2, 'omitnan'); ... 
%         mean(numspikes_ctrl_ep2, 'omitnan'); ...
%         mean(numspikes_opto_ep3, 'omitnan'); ...
%         mean(numspikes_ctrl_ep3, 'omitnan')]','grouped','FaceColor','green');
% hold on
% plot(1,numspikes_opto_ep2,'ok')
% plot(2,numspikes_ctrl_ep2,'ok')
% plot(3,numspikes_opto_ep3,'ok')
% plot(4,numspikes_ctrl_ep3,'ok')
% xticklabels(["opto ep2 5 trials", "control ep2 5 trials", "opto ep3 5 trials", ...
%             "control ep3 5 trials"])
% title(sprintf('animal %s', animal))
% ylabel('total dff/cells/s')