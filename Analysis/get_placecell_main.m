
function [place_cells,total_place_cells,false_cells_num] = get_placecell_main(Fc3,position,nbins, ...
    track_length,Fs)
%%


%%

time_moving = get_moving_time(position,5,Fs);
pos_moving = position(time_moving);
allcellsactivity_fc3 = Fc3(time_moving,:);

[~, actual_place_cells, field_area,peak_bin,~,~,~] =  ...
   get_putative_place_cells(pos_moving, allcellsactivity_fc3,nbins,track_length);

[~,false_idx] = do_bootstrap_shuffle_2p(pos_moving, allcellsactivity_fc3,actual_place_cells,nbins,track_length);

false_cells_num = length(false_idx);
actual_place_cells(false_idx) = [];
field_area(false_idx) = [];
peak_bin(false_idx) = [];
clear peak_bin_sorted sorted_place_cells actual_place_cells_sorted
[~,sorted_place_cells] = sort(peak_bin);
actual_place_cells_sorted = actual_place_cells(sorted_place_cells);
field_area_sorted = field_area(sorted_place_cells);
% figure; 
% imagesc(cell_activity_norm(actual_place_cells_sorted,:));
 
 if ~isempty(actual_place_cells_sorted) 
total_place_cells = length(actual_place_cells_sorted);
place_cells = actual_place_cells_sorted;

 else 
     total_place_cells = 0;
place_cells = [];

 end 