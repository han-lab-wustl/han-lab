clear all;
load('Y:\sstcre_analysis\celltrack\e145_week01-02_plane1\Results\commoncells_atleastoneactivedayperweek_4weeks_week2daymap.mat')
src = 'Y:\sstcre_analysis\'; % main folder for analysis
animal = 'e145';%'e200';%e200';
% plane = 2; % if necessary
% load mats from all days
% fls = dir(fullfile(src, "fmats",animal, 'days', '*day*_Fall.mat'));%dir('Z:\cellreg1month_Fmats\*YC_Fall.mat');
plane = 0;
fls = dir(fullfile(src, "fmats",animal, 'days', sprintf('plane%i', plane), '*day*_Fall.mat'));%dir('Z:\cellreg1month_Fmats\*YC_Fall.mat');
days = cell(1, length(fls));
for fl=1:length(fls)
    disp(fl);
    dy = fls(fl);
    days{fl} = load(fullfile(dy.folder,dy.name), 'iscell');
end
cc=cellmap2dayacrossweeks;
sessions_total=length(days);

% get iscell per day
ratio = {}; probs_tracked = {}; probs_iscell = {}; probs_untracked = {};
for dd = 1:length(days)
    disp(dd)
    cellind = cc(:,dd);
    cellind = cellind(cellind>0);
    iscellind = days{dd}.iscell(cellind,:);
    iscelltotal = sum(days{dd}.iscell(:,1));
    iscelltracked = sum(iscellind(:,1));
    ratio{dd} = iscelltracked/iscelltotal;
    probs_iscell{dd} = days{dd}.iscell(:,2);
    probs_untracked{dd} = days{dd}.iscell(setdiff(1:length(days{dd}.iscell), cellind),2);
    probs_tracked{dd} = iscellind(:,2);    
end

% get average of iscell prob across days
 
figure; 
histogram(cell2mat(ratio))
figure; 
histogram(cell2mat(cellfun(@(x) x', probs_tracked, 'UniformOutput', false)), 'NumBins', 10); hold on; 
histogram(cell2mat(cellfun(@(x) x', probs_iscell, 'UniformOutput', false)), 'NumBins', 10)

sctt = cell2mat(cellfun(@(x) x', probs_tracked, 'UniformOutput', false))';
% scatter(ones(1, length(sct)), sct, 'Jitter', 'on'); hold on; 
sctut = cell2mat(cellfun(@(x) x', probs_untracked, 'UniformOutput', false))';
sctall = cell2mat(cellfun(@(x) x', probs_iscell, 'UniformOutput', false))';
g1 = repmat({'tracked'},length(sctt),1);
g2 = repmat({'untracked'},length(sctut),1);
g3 = repmat({'all'},length(sctall),1);
g = [g1;g2;g3];
x =[sctt; sctut; sctall];
figure; 

boxplot(x,g)
ylabel('suite2p classifier probability')
% scatter(ones(1, length(sct))*2, sct, 'Jitter', 'on'); 
