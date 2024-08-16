function output = deinterleave_array(input_array, num_columns)
    % Check if the length of the input array is divisible by num_columns
    if mod(length(input_array), num_columns) ~= 0
        error('The length of the input array must be divisible by the number of columns.');
    end

    % Calculate the number of rows
    num_rows = length(input_array) / num_columns;

    % Reshape the array into a matrix with the specified number of columns
    reshaped_array = reshape(input_array, num_columns, num_rows)';

    % Output the reshaped array
    output = reshaped_array;
end
