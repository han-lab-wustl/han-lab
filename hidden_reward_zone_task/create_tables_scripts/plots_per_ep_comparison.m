function [comparison_type, tuning_curves_for_this_comparison,...
                real_CS,shuffled_CS,shuffled_CS_cell_index,...
            real_distribution_count,shuffled_distribution_count,...
            RankSumP,RankSumH,RankSumSTATS] = plots_per_ep_comparison(epochs2compare, ...
    this_comparison, tuning_curves, tuning_curves_for_this_comparison, numIterations, N_cells, ...
    comparison_type, shuffled_CS, real_CS, shuffled_CS_cell_index, ...
    shuffled_distribution_count, real_distribution_count, ...
    RankSumP, RankSumH, RankSumSTATS, mouse_cd, n_trials2compare, UL_track, ...
    reward_locations,Settings)

% zahra's adaptation of ele's plots
% zahra uses fc3
EP1 = epochs2compare(this_comparison, 1);
EP2 = epochs2compare(this_comparison, 2);
for this_cell = 1:N_cells
    x = tuning_curves{EP1 }(this_cell,:) ;
    y = tuning_curves{EP2 }(this_cell,:) ;

    cs = getCosineSimilarity(x,y);
    if cs<0
        keyboard
    else
        real_CS{this_comparison}(this_cell,1) = cs;
        shuffled_CS{this_comparison}{this_cell,1} = nan(1,numIterations);
    end

    for i = 1:numIterations
        random_comparison_cell_index = randperm(N_cells,1);
        random_y = tuning_curves{EP2} (random_comparison_cell_index,:);
        cs = getCosineSimilarity(x,random_y);
        if cs<0
            keyboard
        else                        
            shuffled_CS{this_comparison}{this_cell,1}(i) = cs;
            shuffled_CS_cell_index{this_comparison}{this_cell,1}(i) = ...
            random_comparison_cell_index;
        end
    end

    random_cs = shuffled_CS{this_comparison}{this_cell,1};
    real_cs = real_CS{this_comparison}(this_cell,1);
    p(this_cell) = sum(random_cs > real_cs)\numIterations ;
end

comparison_type(this_comparison,:) = [EP1 EP2];
tuning_curves_for_this_comparison{this_comparison,1} = tuning_curves{EP1};
tuning_curves_for_this_comparison{this_comparison,2} = tuning_curves{EP2};
% imagesc plot
[~,max_bin1] = max(tuning_curves_for_this_comparison{this_comparison,1},[],2);
[~,max_bin2] = max(tuning_curves_for_this_comparison{this_comparison,2},[],2);
[~,sorted_idx] = sort(max_bin1);

TC_imagesc = [tuning_curves_for_this_comparison{this_comparison,1}(sorted_idx,:) ...
    tuning_curves_for_this_comparison{this_comparison,2}(sorted_idx,:)];
% ----

shuffled_distribution = cell2mat(shuffled_CS{this_comparison});
shuffled_distribution_count{this_comparison,1} = histcounts...
    (shuffled_distribution,'Normalization','probability','BinWidth',0.025);

real_distribution = real_CS{this_comparison};
real_distribution_count{this_comparison,1} = histcounts...
    (real_distribution,'Normalization','probability','BinWidth',0.025);
% [P,H,STATS] = ranksum(real_distribution_count{this_comparison,1},shuffled_distribution_count{this_comparison,1});
try
    [P,H,STATS] = ranksum(real_distribution,reshape(shuffled_distribution, ...
        [1, numel(shuffled_distribution)]));
catch
    keyboard
end
RankSumP(this_comparison,1) = P;
RankSumH(this_comparison,1) = H;
RankSumSTATS{this_comparison,1} = STATS;

figure('Renderer', 'painters', 'Position', [20 20 1000 700])
fig = tiledlayout('flow');
nexttile
hold on

h1 = histogram(shuffled_distribution,'Normalization','probability','BinWidth',0.025);
h1.FaceColor = [0 0 0];
h1.EdgeColor = [1 1 1];
mean_shuf = mean(mean(shuffled_distribution,'omitnan'),'omitnan');
xline(mean_shuf,'-k')

h2= histogram(real_distribution,'Normalization','probability','BinWidth',0.025);
h2.FaceColor = [1 0 0];
h2.EdgeColor = [1 1 1];
mean_real = mean(real_distribution,'omitnan');
xline(mean_real,'-r')

axis padded
%                             axis off

title(fig,[string(mouse_cd)],[ 'ep' num2str(EP1) ' vs. ep' num2str(EP2) '; RankSum p = ' ...
    num2str(P) '; last ' num2str(n_trials2compare) ' trials comparison'])
legend({'shuffled',['mean shuffle = ' num2str(mean_shuf)],...
    'real',['mean real = ' num2str(mean_real)]},'Location','northwest')

this = nexttile;

imagesc(normalize(TC_imagesc,2))%imagesc(normalize(TC_imagesc,2,'range'))
colormap(turbo)
xlabel('cm')
xticks([0 max([max_bin1; max_bin2])])
xticklabels([0 UL_track])
title(this, 'tuning curve last 8 trials',{['epoch ' num2str(EP1) ';epoch ' num2str(EP2)]})

axis square

this = nexttile;

histogram(max_bin1)
hold on
histogram(max_bin2)
title(this, 'Distribution of Max Bin')
legend({['epoch ' num2str(EP1)],['epoch ' num2str(EP2)]})
ylabel('cell count')
xlabel('cm')
xticks([0 max([max_bin1; max_bin2])])
xticklabels([0 UL_track])
axis padded
axis square

this = nexttile;
title(this, 'Max Peak shift')
parallelcoords([max_bin1,max_bin2],'Color',[.8 .8 .8])
hold on
parallelcoords([max_bin1,max_bin2],'quantile',.25,'LineWidth',2)
yticks([0 max([max_bin1; max_bin2])])
xticklabels({num2str(EP1),num2str(EP2)})
yticklabels([0 UL_track])

xlabel('Epoch')
ylabel('Max Peak')
axis padded
axis square

this = nexttile;

title(this, 'Max Peak Correlation')
scatter(max_bin1*Settings.bin_size,max_bin2*Settings.bin_size,'ko')
xline(reward_locations(EP1),'--r', 'LineWidth',5)
yline(reward_locations(EP2),'--r', 'LineWidth',5)
xlabel('max peak EP1')
ylabel('max peak EP2')

axis padded
axis square

drawnow

saveas(fig,[Settings.saving_path num2str(n_trials2compare) 'trials_CS_histograms_' char(mouse_cd(1:end-9)) ...
    '_ep' num2str(EP1) '_vs_ep' num2str(EP2)],'fig')

close all % afrer saving
end

