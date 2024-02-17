function [real_distribution, shuffled_distribution] = get_tuning_curve_dist(tuningcurve1, tuningcurve2)

numIterations = 1000; % same as ele
n_cells = size(tuningcurve1,1);
for this_cell = 1:n_cells
    x = tuningcurve1(this_cell,:) ;
    y = tuningcurve2(this_cell,:) ;
    cs = corrcoef(x,y);
    cs = cs(1,2);
    % if getCosineSimilarity(x,y)<0 % had to move this to 'Analysis'
    %     cs = 0; % zd changed
    % else
    %     cs = getCosineSimilarity(x,y);
    % end
        real_CS(this_cell,1) = cs;
        shuffled_CS{this_cell,1} = nan(1,numIterations);    

    for i = 1 : numIterations
        random_comparison_cell_index = randperm(n_cells,1);
        random_y = tuningcurve2(random_comparison_cell_index,:);
        cs = corrcoef(x,random_y);
        cs = cs(1,2);
        % if getCosineSimilarity(x,y)<0
        %     cs=0;
        % else
        %     cs = getCosineSimilarity(x,random_y);            
        % end
        shuffled_CS{this_cell,1}(i) = cs;
        shuffled_CS_cell_index{this_cell,1}(i) = random_comparison_cell_index;
    end

    random_cs = shuffled_CS{this_cell,1};
    real_cs = real_CS(this_cell,1);
    p(this_cell) = sum(random_cs > real_cs)\numIterations ;
end


shuffled_distribution = cell2mat(shuffled_CS);
% shuffled_distribution_count = histcounts...
%     (shuffled_distribution,'Normalization','probability','BinWidth',0.025);

real_distribution = real_CS;
% real_distribution_count = histcounts...
%     (real_distribution,'Normalization','probability','BinWidth',0.025);

end