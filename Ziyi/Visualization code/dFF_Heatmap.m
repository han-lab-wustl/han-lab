% Initialize storage for all data
allData = [];

% Repeatedly ask the user to pick files
while true
    [files, path] = uigetfile('*.mat', 'Select data files', 'MultiSelect', 'on');
    if isequal(files, 0)
        % Check if user pressed cancel
        if isempty(allData)
            disp('No files selected. Exiting.');
            return;  % Exit if no files were ever selected
        else
            break;  % Exit loop if user cancels after selecting some files
        end
    end
    
    if iscell(files)  % Check if multiple files are selected
        numFiles = length(files);
    else
        numFiles = 1;
        files = {files};  % Convert to cell array for consistency
    end

    % Read data from each selected file
    for i = 1:numFiles
        fullPath = fullfile(path, files{i});
        loadedData = load(fullPath);  % Load the .mat file
        % Assume the variable of interest is named 'dataMatrix'
        if isfield(loadedData, 'params')
            data = loadedData.dataMatrix;
            allData = [allData; data];  % Concatenate data vertically
        else
            error('Variable "dataMatrix" not found in %s.', files{i});
        end
    end
end