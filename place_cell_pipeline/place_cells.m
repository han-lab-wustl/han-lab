%% Place cells analysis for all cells
function [place_cell_num , actual_place_cells, field_area,...
    peak_bin,pf_new,cell_activity_smoothed,...
    cell_activity_smoothed_norm] = place_cells(time_moving, position, allcellsactivity_fc3 ) %moving position and moving all cells activity

% allcellsactivity_fc3 = calc_Fc3(dFF_new);
% position = linmap(y2(:,1), [0, 180]);

allcellsactivity_fc3(isnan(allcellsactivity_fc3)) = 0;
track_length = 270;
bin_size = 3;
nbins = track_length/bin_size;

% position = pre.ypos; 
% allcellsactivity_fc3 = pre.Fc3_all;
% find y position during positive velocity 
pos_moving = position(time_moving);

for i = 1:nbins  
    time_in_bin{i} = time_moving((pos_moving >= (i-1)*bin_size) & (pos_moving < i*bin_size));  
end 

% BINNED cell activity
moving_cells_activity = allcellsactivity_fc3(:,time_moving);
for i = 1:size(allcellsactivity_fc3,1)
    for bin = 1:nbins
        cell_activity(i,bin) = mean(allcellsactivity_fc3(i,time_in_bin{bin})); 
    end
end 
cell_activity(isnan(cell_activity)) = 0;
cell_activity_smoothed = smoothdata(cell_activity,2,'gaussian',5);

for i = 1:size(cell_activity_smoothed,1)
    cell_activity_smoothed_norm(i,:) =  cell_activity_smoothed(i,:)/max(cell_activity_smoothed(i,:));
end 
cell_activity_smoothed_norm(isnan(cell_activity_smoothed_norm)) = 0;
for i = 1:size(allcellsactivity_fc3,1)
    [peak_val(i),peak_val_bin(i)] = max(cell_activity_smoothed(i,:));
    base_val(i) = baseline_val(cell_activity_smoothed(i,:),15);
    peak_base_diff(i) =  peak_val(i) - base_val(i);    
end 

%Detecting infield for cells 
place_field = zeros(size(allcellsactivity_fc3,1),nbins);
for neuron = 1:size(allcellsactivity_fc3,1)
    % neuron = 579;
    [~,place_bin] = max(cell_activity_smoothed(neuron,:));
    start_bin = 0;
    while(cell_activity_smoothed(neuron,place_bin-start_bin) >= base_val(neuron) + 0.25*peak_base_diff(neuron))
        if start_bin < place_bin
        start_bin = start_bin + 1;
        end
        
        if start_bin == place_bin
            break; 
        end 
    end
    
    if start_bin ~= place_bin
    start_bin = place_bin - start_bin+1;
    else
        start_bin = place_bin;
    end
    
    end_bin = 0;
    while(cell_activity_smoothed(neuron,place_bin+end_bin) >= base_val(neuron) + 0.25*peak_base_diff(neuron))
        if place_bin+end_bin < nbins
       end_bin = end_bin + 1;
        end
        
        if place_bin+end_bin == nbins
            break; 
        end 
    end
    
    if place_bin+end_bin ~= nbins
        end_bin = place_bin + end_bin -1;
    else
        end_bin = nbins;
    end
    place_field(neuron,start_bin:end_bin) = 1;
end
alpha = num_place_cells(place_field);

% Calculating length of place field - minimum 15cm and eliminating 

for i = 1:size(allcellsactivity_fc3,1)
    field_length = length(find(place_field(i,:) == 1));
    if field_length < 3
        place_field(i,:) = zeros(1,nbins);
    end  
end 
alpha = num_place_cells(place_field);

% Mean infield dF/F values for each cell
pf_old = place_field; 
for i = 1:size(allcellsactivity_fc3,1)
    
    field_area = find(place_field(i,:) == 1); 
    if isempty(field_area)
         mean_infield_df_f(i) = 0;
    else
        mean_infield_df_f(i) = mean(cell_activity_smoothed(i,field_area));   
    end 
    
    out_field_area = place_field(i,:) == 0;
    mean_out_field_df_f(i) = mean(cell_activity_smoothed(i,out_field_area));
    
        if mean_infield_df_f(i) < 3*mean_out_field_df_f(i)
            place_field(i,:) = zeros(1,nbins);
        end
   
    clearvars field_area out_field_area ;  
        
end

alpha = num_place_cells(place_field);
pf_new = place_field;
alpha1 = num_place_cells(pf_new);

for i = 1:size(allcellsactivity_fc3,1)
        num_events(i) = 0;
        time_in_field(i) = 0;
        field_area = find(place_field(i,:) == 1); 
        if ~isempty(field_area)
            for field = field_area(1) : field_area(length(field_area))
                num_events(i) = num_events(i) + length(find(allcellsactivity_fc3(i,time_in_bin{field}) ~=0 ));
                time_in_field(i) = time_in_field(i) + length((allcellsactivity_fc3(i,time_in_bin{field})));
            end
            percent_of_time_active(i) = num_events(i)*100/time_in_field(i);
        else 
            percent_of_time_active(i) = 0;
        end 
        if percent_of_time_active(i) < 1
            pf_new(i,:) = zeros(1,nbins);
        end
        clearvars field_area peak_bin
end 
alpha = num_place_cells(pf_new);
[place_cell_num, actual_place_cells] = num_place_cells(pf_new);

% figure; 
% histogram(percent_of_time_active,5)

if place_cell_num == 0 
     field_area = 0;
     peak_bin = 0;
else
for i = 1:place_cell_num
    field_area{i} = find(pf_new(actual_place_cells(i),:) == 1);
end 
for i = 1:place_cell_num
    
   [~,peak_bin(i)] = max(cell_activity_smoothed(actual_place_cells(i),:));
    
end 

end

