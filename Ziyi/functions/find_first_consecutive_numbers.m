function first_numbers = find_first_consecutive_numbers(arr)
    % Initialize the output array to store the first numbers of consecutive sequences
    first_numbers = [];

    % Check if the array is empty
    if isempty(arr)
        return;
    end

    % Add the first element of the array to the output, as it is the start of the first sequence
    first_numbers(end+1) = arr(1);

    % Loop through the array to find the start of each consecutive sequence
    for i = 2:length(arr)
        if arr(i) ~= arr(i-1) + 1
            % If the current number is not consecutive to the previous number, it is the start of a new sequence
            first_numbers(end+1) = arr(i);
        end
    end
end