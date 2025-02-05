function stretches = findConsecutiveNonZeroIndices(arr)
    % This function returns the start and end indices of consecutive
    % stretches of non-zero integers in the input array `arr`.
    % Additionally, stretches need to be at least 100 frames apart.

    if isempty(arr)
        stretches = [];
        return;
    end

    % Find the indices where the array is non-zero
    nonZeroIndices = find(arr ~= 0);

    if isempty(nonZeroIndices)
        stretches = [];
        return;
    end

    % Initialize the output
    stretches = [];
    
    % Initialize the start of the first stretch
    startIdx = nonZeroIndices(1);
    lastEndIdx = -inf;  % Initialize to a very small value to handle the first stretch

    for i = 2:length(nonZeroIndices)
        % If the current index is not consecutive
        if nonZeroIndices(i) ~= nonZeroIndices(i-1) + 1
            % End the current stretch
            endIdx = nonZeroIndices(i-1);
            
            % Check if the current stretch is at least 100 frames apart from the last one
            if startIdx - lastEndIdx >= 100
                stretches = [stretches; startIdx, endIdx];
                lastEndIdx = endIdx;
            end
            
            % Start a new stretch
            startIdx = nonZeroIndices(i);
        end
    end

    % End the final stretch
    if startIdx - lastEndIdx >= 100
        stretches = [stretches; startIdx, nonZeroIndices(end)];
    end
end
