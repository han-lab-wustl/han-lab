

function cell_sparsity = get_spatial_sparsity_all_cells(Fc3,position,Fs,nBins,track_length)
%% Fc3 = dFF of all cells in N X T format where N - number of cells and T - time  
% position - position of animal on track 
% Fs - Frame rate of acquisition
% nBins - number of bins in which you want to divide the track into 
% track_length - Length of track

%%

nCells = size(Fc3,2);
for cell = 1:nCells
    cell_sparsity(cell) = get_spatial_sparsity_per_cell(Fc3(:,cell),position,Fs,nBins,track_length);
end


