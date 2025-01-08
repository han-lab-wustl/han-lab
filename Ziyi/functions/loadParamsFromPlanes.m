function paramsData = loadParamsFromPlanes(mainFolder)
    % Get all folders inside the main directory matching suite2p
    dir_s2p = struct2cell(dir(fullfile(mainFolder, '**', 'suite2p')));
    
    % Filter folders containing the word 'plane' in their name
    planefolders = dir_s2p(:, ~cellfun(@isempty, regexp(dir_s2p(1, :), 'plane')));
    
    % Initialize a structure to store the params.mat for each plane
    paramsData = struct();
    
    % Loop through each plane folder
    for i = 1:size(planefolders, 2)
        planeFolderPath = fullfile(planefolders{2, i}, planefolders{1, i});
        
        % Define the path to the 'reg_tif' folder inside the plane folder
        regTifFolderPath = fullfile(planeFolderPath, 'reg_tif');
        
        % Define the path to the params.mat file inside the 'reg_tif' folder
        paramsFile = fullfile(regTifFolderPath, 'params.mat');
        
        % Check if params.mat exists in the 'reg_tif' folder
        if exist(paramsFile, 'file')
            % Load the params.mat file
            params = load(paramsFile);
            
            % Extract plane name for storing in the structure
            [~, planeFolderName] = fileparts(planefolders{1, i});
            
            % Store the params in a structure field named after the plane folder
            paramsData.(planeFolderName) = params;
        else
            warning('params.mat not found in reg_tif folder: %s', regTifFolderPath);
        end
    end
end