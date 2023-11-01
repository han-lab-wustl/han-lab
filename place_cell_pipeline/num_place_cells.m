function [y,z] = num_place_cells(X)
   k = max(X');
   y = length(find(k == 1));
   z = find(k == 1);
end 