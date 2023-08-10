function [dff] = remove_border_cells(stat, Falliscell, F, ...
    cell2remove, remove_iscell, all)
% zd wrote for opto experiments to remove border cells from top and bottom
% borders
% get xy axis
    stat_iscell = stat(logical(Falliscell(:,1)));
    if ~(size(stat_iscell,2)==size(F,1)) % check if same size as all.dff (sometimes cells are not removed) 
        if exist('cell2remove', 'var') % check if cell2remove var exists
            stat_cell2remove = stat_iscell(~logical(cell2remove)&(~logical(remove_iscell)));
        else
            stat_cell2remove = stat_iscell((~logical(remove_iscell)));
        end
    else
        stat_cell2remove = stat_iscell;
    end
    Ypix = cellfun(@(x) x.ypix, stat_cell2remove, 'UniformOutput', false);
    bordercells = zeros(1,length(Ypix)); % bool of cells at the top border
    for yy=1:length(Ypix) % idea is to remove these cells
        if sum(Ypix{yy}<100)>0 || sum(Ypix{yy}>460)>0 % crop bottom border cells (image dim 512)
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

    dff = all.dff(~logical(bordercells),:); 
end