load('cs_table_last8trials.mat')

%% compare shhuffled vs real remap values 
real_CS_distribution = cell2mat(cs_table.real_CS);
shuffled_CS_distribution_cell = cellfun(@(x) cell2mat(x), cs_table.shuffled_CS, 'UniformOutput',false) ; 
shuffled_CS_distribution = cell2mat(shuffled_CS_distribution_cell);

figure;
histogram(real_CS_distribution,"Normalization","probability","BinWidth",0.05)
hold on
histogram(shuffled_CS_distribution,"Normalization","probability","BinWidth",0.05)
legend({'real','shuffled'},"Location","northwest")
title( 'cosine similarity values real vs shuffled' ,'all days all comparisons pulled together')


%%
% close all
%  uiopen('/media/gaia/Elements/E186/Summary.xlsx',1) % import excell summary
%Summary. Properties. VariableNames{2} = 'experiment'; % change var name
fam_days_idx = find(strcmp(Summary.experiment,'fam'));% fam days only
fam_days = Summary.E186(fam_days_idx);
fam_days=string(fam_days);

indeces = zeros(1,height(cs_table));
for i = 1 : numel(fam_days)
    indeces(cs_table.Day == fam_days(i)) = true;
end
fam_days_idx_epoch = zeros(1,height(cs_table));
fam_days_idx_epoch(cs_table.comparison_type(:,1)==1 & cs_table.comparison_type(:,2)==2) = 1;
indeces = indeces + fam_days_idx_epoch;
indeces = indeces == 2;


only_fam_dist = cellfun(@(x) x.ranksum, cs_table.RankSumSTATS(indeces==1), 'UniformOutput',false) ;
contidion = 'remap1';
remap_days_idx = find(strcmp(Summary.experiment,contidion));% fam days only
remap_days = Summary.E186(remap_days_idx);

% hist corr
% compare shuffled vs real remap values 
real_CS_distribution = cell2mat(cs_table.real_CS(indeces==1));
shuffled_CS_distribution_cell = cellfun(@(x) cell2mat(x), cs_table.shuffled_CS(indeces==1), 'UniformOutput',false) ; 
shuffled_CS_distribution = cell2mat(shuffled_CS_distribution_cell);

figure;
histogram(real_CS_distribution,"Normalization","probability","BinWidth",0.05)
hold on
histogram(shuffled_CS_distribution,"Normalization","probability","BinWidth",0.05)
legend({'real','shuffled'},"Location","northwest")
title( 'real CS vs Shuffle' ,'only FAMILIAR epoc1 vs epoc2')

indeces = zeros(1,height(cs_table));
for i = 1 : numel(remap_days)
    indeces(cs_table.Day == remap_days{i}) = 1;
end

remap_days_idx_epoch = zeros(1,height(cs_table));
remap_days_idx_epoch(cs_table.comparison_type(:,1)==1 & cs_table.comparison_type(:,2)==2) = 1;
indeces = indeces + remap_days_idx_epoch;
indeces = indeces == 2;

only_remap_dist = cellfun(@(x) x.ranksum, cs_table.RankSumSTATS(indeces==1), 'UniformOutput',false) ; 

% hist corr
% compare shuffled vs real remap values 
real_CS_distribution = cell2mat(cs_table.real_CS(indeces==1));
shuffled_CS_distribution_cell = cellfun(@(x) cell2mat(x), cs_table.shuffled_CS(indeces==1), 'UniformOutput',false) ; 
shuffled_CS_distribution = cell2mat(shuffled_CS_distribution_cell);

figure;
histogram(real_CS_distribution,"Normalization","probability","BinWidth",0.05)
hold on
histogram(shuffled_CS_distribution,"Normalization","probability","BinWidth",0.05)
legend({'real','shuffled'},"Location","northwest")
title( 'real CS vs Shuffle' ,'only REMAP epoc1 vs epoc2')

% hist stat
figure;
histogram(cell2mat(only_fam_dist),"Normalization","probability","BinWidth",9^8)
hold on
histogram(cell2mat(only_remap_dist),"Normalization","probability","BinWidth",9^8)
legend({'fam','remap'})
title( 'KS stat values - histogram')

% [h,p] = ttest2(cell2mat(only_fam_dist),cell2mat(only_remap_dist));
% [h,p] = kstest2(cell2mat(only_fam_dist),cell2mat(only_remap_dist));
% [p,h] = ranksum(cell2mat(only_fam_dist),cell2mat(only_remap_dist));

fam_values = cell2mat(only_fam_dist);
remap_values = cell2mat(only_remap_dist);
fam_labels  = repmat({'fam'},size(fam_values));
remap_labels = repmat({'remap'},size(remap_values));

figure;
boxplot([fam_values; remap_values],[fam_labels; remap_labels])
ylabel('KS test')
title( 'KS stat values - boxplot')
%% different experimental condition corr comparison

fam_days_idx = find(strcmp(Summary.experiment , 'fam'));% fam days only
fam_days = Summary.E186(fam_days_idx);
fam_days=string(fam_days);

indeces = zeros(1,height(cs_table));
for i = 1 : numel(fam_days)
    indeces(cs_table.Day == fam_days(i)) = true;
end
fam_days_idx_epoch = zeros(1,height(cs_table));
fam_days_idx_epoch(cs_table.comparison_type(:,1)==1) = 1;
indeces = indeces + fam_days_idx_epoch;
indeces = indeces == 2;

only_fam_dist = cellfun(@(x) x, cs_table.real_CS(indeces==1), 'UniformOutput',false) ;
only_fam_dist_vector = cell2mat(only_fam_dist);


remap_days_idx = find(strcmp(Summary.experiment , 'remap1') ); % | Summary.experiment == 'remap2');% fam days only
remap_days = Summary.E186(remap_days_idx);

indeces = zeros(1,height(cs_table));
for i = 1 : numel(remap_days)
    indeces(cs_table.Day == remap_days{i}) = 1;
end

remap_days_idx_epoch = zeros(1,height(cs_table));
remap_days_idx_epoch(cs_table.comparison_type(:,1)==1) = 1;
indeces = indeces + remap_days_idx_epoch;
indeces = indeces == 2;

only_remap_dist = cellfun(@(x) x, cs_table.real_CS(indeces==1), 'UniformOutput',false) ;
only_remap_dist_vector = cell2mat(only_remap_dist);

remap2_days_idx = find(strcmp(Summary.experiment , 'remap2' )); % | Summary.experiment == 'remap2');% fam days only
remap2_days = Summary.E186(remap2_days_idx);

indeces = zeros(1,height(cs_table));
for i = 1 : numel(remap2_days)
    indeces(cs_table.Day == remap2_days{i}) = 1;
end

remap2_days_idx_epoch = zeros(1,height(cs_table));
remap2_days_idx_epoch(cs_table.comparison_type(:,1)==1) = 1;
indeces = indeces + remap2_days_idx_epoch;
indeces = indeces == 2;

only_remap2_dist = cellfun(@(x) x, cs_table.real_CS(indeces==1), 'UniformOutput',false) ;
only_remap2_dist_vector = cell2mat(only_remap2_dist);


figure;
hold on
%h3 = histogram(only_remap2_dist_vector,"Normalization","probability","BinWidth",0.05,'FaceColor','green',"EdgeColor","w","FaceAlpha",0.6);
h2 = histogram(only_remap_dist_vector,"Normalization","probability","BinWidth",0.05,'FaceColor','blue',"EdgeColor","w","FaceAlpha",0.6);
h1 = histogram(only_fam_dist_vector,"Normalization","probability","BinWidth",0.05,"FaceColor",'red',"EdgeColor","w","FaceAlpha",0.6);

if  numel(remap_days)>0

xline(median(only_remap_dist_vector),'-b');
end
xline(median(only_fam_dist_vector),'-r');


legend({'remap1','fam',['median remap = ' num2str(median(only_remap_dist_vector))],['median fam = ' num2str(median(only_fam_dist_vector))]},"Location","northwest")
title( 'CS values FAM vs REMAP1')

text(min(xlim),max(ylim)/2,['RankSum p = ' num2str(ranksum(only_fam_dist_vector,only_remap_dist_vector,'tail','right'))]...
,'FontSize',12)

axis padded


figure;
hold on
%h3 = histogram(only_remap2_dist_vector,"Normalization","probability","BinWidth",0.05,'FaceColor','green',"EdgeColor","w","FaceAlpha",0.6);
h2 = histogram(only_remap2_dist_vector,"Normalization","probability","BinWidth",0.05,'FaceColor','green',"EdgeColor","w","FaceAlpha",0.6);
h1 = histogram(only_fam_dist_vector,"Normalization","probability","BinWidth",0.05,"FaceColor",'red',"EdgeColor","w","FaceAlpha",0.6);

xline(median(only_remap2_dist_vector),'-g');
xline(median(only_fam_dist_vector),'-r');


legend({'remap2','fam',['median remap = ' num2str(median(only_remap2_dist_vector))],['median fam = ' num2str(median(only_fam_dist_vector))]},"Location","northwest")
title( 'CS values FAM vs REMAP2')

text(min(xlim),max(ylim)/2,['RankSum p = ' num2str(ranksum(only_fam_dist_vector,only_remap2_dist_vector,'tail','right'))]...
,'FontSize',12)

axis padded