function indices = find_indices_with_consecutive_before(arr)
    % Initialize the output array to store the indices of elements with at least 2 consecutive numbers before them
    indices = [];

    % Check if the array has less than 3 elements (in which case there cannot be 2 consecutive numbers before any element)
    if length(arr) < 3
        return;
    end

    % Loop through the array starting from the third element
    for i = 3:length(arr)
        if arr(i-1) == arr(i-2) + 1 && arr(i) == arr(i-1) + 1
            % If the current element and the two previous elements are consecutive, add the index to the output
            indices(end+1) = i;
        end
    end
end