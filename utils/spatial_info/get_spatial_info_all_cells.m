

function cell_info = get_spatial_info_all_cells(Fc3,fv,thres, ftol, ...
    position,Fs,nBins,track_length)
%% Fc3 = dFF of all cells in N X T format where N - number of cells and T - time  
% position - position of animal on track 
% Fs - Frame rate of acquisition
% nBins - number of bins in which you want to divide the track into 
% track_length - Length of track

%%

nCells = size(Fc3,1); % zd fixed the order because the description did not match the input order of the matrix, 7/4/23
for cell = 1:nCells
    cell_info(cell) = get_spatial_info_per_cell(Fc3,fv,thres, ftol, ...
    position,Fs,nBins,track_length);
end


