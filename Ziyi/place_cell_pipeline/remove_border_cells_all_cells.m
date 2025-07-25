function [dFF,bordercells] = remove_border_cells_all_cells(stat, dFF)
% zd wrote for opto experiments to remove border cells from top and bottom
% borders
% get xy axis
    
    Ypix = cellfun(@(x) x.ypix, stat, 'UniformOutput', false);
    bordercells = zeros(1,length(Ypix)); % bool of cells at the top border
    for yy=1:length(Ypix) % idea is to remove these cells
        if sum(Ypix{yy}<100)>0 || sum(Ypix{yy}>460)>0 % crop top and bottom border cells (image dim 512)
            bordercells(yy)=1;
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
    bordercells = logical(bordercells);
    dFF(:,bordercells) = NaN;
end