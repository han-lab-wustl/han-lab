% Define the main folder where 'suite2p' is located
mainFolder = 'E:\Ziyi\Data\241008_ZH\241008_ZH_000_000'; % Change this to your actual path

% Get the list of subdirectories inside the main folder
suite2pFolder = fullfile(mainFolder, 'suite2p');
if ~isfolder(suite2pFolder)
    error('suite2p folder not found.');
end

% Get a list of all plane folders inside the 'suite2p' folder
planeFolders = dir(suite2pFolder);
planeFolders = planeFolders([planeFolders.isdir] & ~ismember({planeFolders.name}, {'.', '..'})); % Filter out non-folders

% Initialize a structure to store the params.mat for each plane
paramsData = struct();

% Loop through each plane folder
for i = 1:length(planeFolders)
    planeFolderName = planeFolders(i).name;
    planeFolderPath = fullfile(suite2pFolder, planeFolderName);
    
    % Define the path to the 'reg_tif' folder inside the plane folder
    regTifFolderPath = fullfile(planeFolderPath, 'reg_tif');
    
    % Check if the 'reg_tif' folder exists
    if isfolder(regTifFolderPath)
        % Define the path to the params.mat file inside the 'reg_tif' folder
        paramsFile = fullfile(regTifFolderPath, 'params.mat');
        
        % Check if params.mat exists in the 'reg_tif' folder
        if exist(paramsFile, 'file')
            % Load the params.mat file
            params = load(paramsFile);
            
            % Store the params in a structure field named after the plane folder
            paramsData.(planeFolderName) = params;
        else
            warning('params.mat not found in reg_tif folder: %s', regTifFolderPath);
        end
    else
        warning('reg_tif folder not found in plane folder: %s', planeFolderPath);
    end
end

% Display the loaded params for each plane
disp(paramsData);
