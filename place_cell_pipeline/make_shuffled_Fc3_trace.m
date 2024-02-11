function [shuffled_cells_activity] = make_shuffled_Fc3_trace(bins2shuffle_forcell, ...
    fc3, shuffledbins_forcell, shuffled_cells_activity)
for i = 1:size(fc3,2)
    try
        s = shuffle(bins2shuffle_forcell{i});
        shuffledbins_forcell{i} = s; 
        s_ = s(~cellfun(@isempty,s)); % 5 is the bin limit apprently - why?
        sca = fc3(cell2mat(s_),i);
        shuffled_cells_activity(:,i) = sca;
    catch
        % disp(i)
%         disp('\n likely a cell with no transient \n')
end 
end