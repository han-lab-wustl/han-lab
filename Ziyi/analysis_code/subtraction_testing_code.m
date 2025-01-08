% Define two main folders where the 'suite2p' folders are located
mainFolder1 = 'E:\Ziyi\Data\241008_ZH\241008_ZH_000_000'; % Change this to your actual first path
mainFolder2 = 'E:\Ziyi\Data\241008_ZH\241008_ZH_000_001'; % Change this to your actual second path

plane = ["plane0", "plane1", "plane2", "plane3"];
% Load params.mat files from the first main folder
paramsData1 = loadParamsFromPlanes(mainFolder1);
% Load params.mat files from the second main folder
paramsData2 = loadParamsFromPlanes(mainFolder2);

stims1 = paramsData1.plane0.stims;
stims2 = paramsData2.plane0.stims;

for i = 1:4
    % Get the name of the current plane
    planeName = plane{i};
    
    % Access the data for the current plane
    planeData1 = paramsData1.(planeName);
    planeData2 = paramsData2.(planeName);



    dFF1 = planeData1.params.roibasemean3{1};
    dFF2 = planeData2.params.roibasemean3{1};

    dFF_base_mean1=mean(dFF1);
    dFF1 = dFF1/dFF_base_mean1;

    dFF_base_mean2=mean(dFF1);
    dFF2 = dFF2/dFF_base_mean2;

    
    
    % Display the plane name and data (you can process the data as needed)
    fprintf('Processing data for %s:\n', planeName);
    %disp(planeData);
    
    % Perform any operation on planeData here
end

figure; plot(stims1)