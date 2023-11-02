matt = [putative_pcs{1}; putative_pcs{2}; putative_pcs{3}; putative_pcs{4}];
figure; histogram(sum(matt,1)); xlabel('# of times cell is a place cell across epochs'); 
ylabel('# of cells')
ep_mask = sum(matt,1);
dff = dFF(:, ep_mask>3);
fc3 = Fc3(:, ep_mask>3);
figure; imagesc(normalize(dff',1))
for i=1:10
    if max(dff(:,i))<10 % removes outlier cells
        figure; 
        plot(dff(:,i)); hold on; 
        plot((rewards==1)*2, 'b*')
        yyaxis right
        plot(ybinned)
    end
end

bin_size = 1;
nbins = track_length/bin_size;

for i = 1:nbins    
    time_in_bin{i} = time_moving(ypos_mov >= (i-1)*bin_size & ypos_mov < i*bin_size);
end 

% make bins via suyash method
moving_cells_activity = fc3(putative_pc,time_moving);
fc3_pc = fc3(putative_pc,:);
for i = 1:size(moving_cells_activity,1)
    for bin = 1:nbins
        cell_activity(i,bin) = mean(fc3_pc(i,time_in_bin{bin})); 
    end
end 
cell_activity(isnan(cell_activity)) = 0;
figure; imagesc(normalize(cell_activity))

% sort by max value
for c=1:length(cell_activity)
    f = cell_activity(c,:);
    if sum(f)>0
        [peakval,peakbin] = max(f);
        peak(c) = peakbin;
    else
        peak(c) = 1;
    end
end

[~,sorted_idx] = sort(peak);
figure; imagesc(normalize(cell_activity(sorted_idx,:)))