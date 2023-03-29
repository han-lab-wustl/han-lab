% compile togheted all the CS between epoch to prepare correlation plot

clc
clear

m_folder = 'I:\E139'; 
% folder where to save my data and from where i am gathering my data
addpath(genpath(m_folder));
Path       = m_folder; % wherever you want to search
searchPath = [Path ,'\**\CS_popActLast50.mat']; % Search in folder and subfolders for  *.csv
Files      = dir(searchPath); % Find all files
f = struct2cell(Files);
files = f(2,:)';

for i = 1:numel(files)
path = files{i};
st = strfind(path, '\');
load([path '\CS_popActLast50.mat'])
mouse_day = repmat(string(nname),size(Cs_struct,1),1);
[Cs_struct.('mouse_day')] = mouse_day;

if ~exist('CS_main','var')
CS_main = Cs_struct;
else
CS_main = [CS_main; Cs_struct];
end

 clearvars -except CS_main i files
end

figure;
scatter(abs(CS_main.diff),CS_main.CS,'filled')

