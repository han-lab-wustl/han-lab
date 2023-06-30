function [cellIdx ,maxBin] = get_sorted_cells_idx(cell_activity)
[~,maxBin] = max(cell_activity,[],2);
[~,cellIdx] = sort(maxBin);