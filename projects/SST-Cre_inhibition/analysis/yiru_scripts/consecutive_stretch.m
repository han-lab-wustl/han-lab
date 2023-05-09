function y = consecutive_stretch(x)

% x = [1 2 3 4 5 8 9 10 12 13 14 18 19 20 21 24];
z = diff(x);

break_point = find(z >1);
if isempty(break_point)
    
    y{1} = x; 
    
    return
end
y = cell(1,length(break_point)+1);

if ~isempty(break_point)
    
    
    y{1} = x(1 : break_point(1));
    for i = 2:length(break_point)
        y{i} = x(break_point(i-1)+1 : break_point(i));
    end
    y{length(break_point)+1} = x(break_point(length(break_point))+1 : length(x));
    
    
    
end





