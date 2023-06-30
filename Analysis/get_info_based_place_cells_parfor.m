function place_cells = get_info_based_place_cells_parfor(Fc3,position,Fs,nBins,track_length)

cell_info = get_spatial_info_all_cells(Fc3,position,Fs,nBins,track_length); % nCelss X 1
cell_info_shuffled = get_shuffled_cell_info_parfor(Fc3,position,Fs,nBins,track_length); %nShuffles X nCells

for cell_num = 1:length(cell_info)
    abs_val = length(find(cell_info_shuffled(:,cell_num) < cell_info(cell_num)));
    p_val(cell_num) = 1 - abs_val/size(cell_info_shuffled,1);
end

place_cells = find(p_val <= 0.05); 