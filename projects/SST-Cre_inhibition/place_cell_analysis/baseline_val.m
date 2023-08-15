function y = baseline_val(x,l)

%X = vector; l = lowest how many numbers

z = sort(x); 
h = z(1:l); 
y = mean(h);
