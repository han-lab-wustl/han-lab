
% plot cells with hrz behavior

%e.g.
clear all;
% look at all cells at one
animal = 'E136';
for d=days
    Fmat  = dir(fullfile(sprintf('F:\\%s\\D%i', animal, d), '**', 'plane*', 'Fall.mat'));
    for plane = 1:length(Fmat)
        load(fullfile(Fmat(plane).folder, Fmat(plane).name))
        figure;
        subplot(3,1,1); imagesc(normalize(dFF_iscell,1)); 
        xlim([0 length(ybinned)])
        subplot(3,1,2)
        plot(forwardvel, 'b'); xlim([0 length(ybinned)])
        subplot(3,1,3); 
        plot(ybinned, 'k'); xlim([0 length(ybinned)])
        hold on; 
        plot(find(rewards==1), ybinned(rewards==1), 'go')
        plot(find(licks), ybinned(licks), 'r.')
        xline(find(trialnum<3),'Alpha',0.01)
        sgtitle(sprintf('Day = %i, plane = %i', d, ...
                    plane))
    end
end
%%
animal = 'E136';
corrs = {};
cell_not_corr_d = {};
days = [5]; % compile across all days
for d=days
    Fmat  = dir(fullfile(sprintf('F:\\%s\\D%i', animal, d), '**', 'plane*', 'Fall.mat')); %'G:\\%s\\Day%i
    corrs_day = {};
    for plane=1:length(Fmat)
        load(fullfile(Fmat(plane).folder, Fmat(plane).name))
        for c=1:size(dFF_iscell,1)            
%             subplot(size(dFF_iscell,1)+2,1,c); plot(dFF_iscell(c,:));                         
%             axis off          
%             sgtitle(sprintf('%s \n Day = %i, plane = %i', animal, ...
%                 d, plane))
            forwardvel = forwardvel(1:length(dFF_iscell(c,:))); % force same dim, issue based on old suite2p
            [corr,p] = corrcoef(dFF_iscell(c,:), ...
                smoothdata(forwardvel, 'gaussian'));
            corrs_day{cell} = corr(1,2);
            
            if corr(1,2)<0.2
                cell_not_corr{cell} = dFF_iscell(c,:);
            end
            figure;
            subplot(3,1,1)
            plot(dFF_iscell(c,:));
            subplot(3,1,2)
            plot(forwardvel, 'b'); xlim([0 length(ybinned)])
            xline(find(trialnum<3),'Alpha',0.01)
            subplot(3,1,3); 
            plot(ybinned, 'k'); xlim([0 length(ybinned)])
            hold on; 
            plot(find(rewards==1), ybinned(rewards==1), 'go')
            plot(find(licks), ybinned(licks), 'r.')
            xline(find(trialnum<3),'Alpha',0.01)
            sgtitle(sprintf('Day = %i, plane = %i, cellno. = %i', d, ...
                plane, cell))
        end        
    end
%     figure;
%     subplot(2,1,1)
%     plot(forwardvel, 'b'); xlim([0 length(ybinned)])
%     xline(find(trialnum<3),'Alpha',0.01)
%     subplot(2,1,2); 
%     plot(ybinned, 'k'); xlim([0 length(ybinned)])
%     hold on; 
%     plot(find(rewards==1), ybinned(rewards==1), 'go')
%     plot(find(licks), ybinned(licks), 'r.')
%     xline(find(trialnum<3),'Alpha',0.01)
%     sgtitle(sprintf('Day = %i', d))

    corrs{d} = corrs_day;
    cell_not_corr_d{d} = cell_not_corr;
%     figure; hist(cell2mat(corrs_day));
%     title(sprintf('Day %i \n correlation of dff to forwardvel', ...
%         d))
end
% figure; hist(cell2mat(cellfun(@(x) cell2mat(x), corrs, 'UniformOutput', false)));
% title('correlation of dff to forwardvel, all days')

%%
% align to reward
clear all; close all
animal = 'E135';
srcdir = 'J:\\%s\\Day%i';
% animal = 'E136';
% srcdir = 'H:\\%s\\D%i';
days = [1:4]; % compile across all days
% days = [2:7];
binnedPerirewards_d = {};
range = 8; % s
bin = 0.1; % s
for d=days
    Fmat  = dir(fullfile(sprintf(srcdir, animal, d), '**', 'plane*', 'Fall.mat')); %'G:\\%s\\Day%i
    binnedPerirewards = {};
    for plane=1:length(Fmat)
        load(fullfile(Fmat(plane).folder, Fmat(plane).name))
        [binnedPerireward,allbins,rewdFF,normmeanrewdFF] = perirewardbinnedactivity(dFF_iscell', ...
            rewards,timedFF,range,bin);                
        normmeanrewdFFs{plane} = normmeanrewdFF;        
    end
    [vbinnedPerireward,vallbins,vrewdFF] = perirewardbinnedactivity(forwardvel', ...
            rewards,timedFF,range,bin);
    daycells_rewbin = [normmeanrewdFFs{1} normmeanrewdFFs{2} normmeanrewdFFs{3}]';
%     [~, sorted_idx] = max(daycells_rewbin, [], 1);
%     plt = daycells_rewbin(sorted_idx, :);
    plt = daycells_rewbin;
    figure; imagesc(plt); colorbar
    ylabel('cells across all planes')
    yyaxis right
    plot(vbinnedPerireward, 'k', 'LineWidth',3)
    xline(median(1:range*20),'w-',{'Reward'});
    xticks([0:10:range*20])
    xticklabels([-range:range])
    title(sprintf('animal = %s, day = %i', animal, d))
    xlabel('seconds from reward(s)')
    ylabel('velocity')
    binnedPerirewards_d{d} = binnedPerirewards;    
end
binnedPerirewards_d = binnedPerirewards_d(~cellfun('isempty',binnedPerirewards_d));
allcells_rewbin = cell2mat(cellfun(@(x) [x{1}; x{2}; x{3}]', binnedPerirewards_d, 'UniformOutput', false));
