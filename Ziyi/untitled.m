% Original interleaved array (example with a smaller size for demonstration)
A = (1:40000)';

% Desired number of columns
num_columns = 4;

% Deinterleave the array
B = deinterleave_array(A, num_columns);

% Display the result (first few rows for brevity)
disp(B(1:10, :));
