% Specify the file name of the .csv file
csvFileName = 'F:\A_\Plot_Values_red_plane4';

% Read the .csv file starting from row 2 and column 2
data = readmatrix(csvFileName, 'Range', 'B2');

% Convert the data to a 1xlength array
rawFluo = data(:)';

% Specify the name of the .mat file
matFileName = 'params.mat';

% Save the variable rawFluo to the .mat file
save(matFileName, 'rawFluo');