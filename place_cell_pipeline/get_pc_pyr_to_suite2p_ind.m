% Zahra - june 2024
% NEED TO RUN FOR CELL TRACKING AND TO GET PLACE CELL TUNING CURVES
% this run script mostly makes plots but calls other functions
% add han-lab and han-lab-archive repos to path! 
clear all; 
% anms = ["e218", "e216", "e217", "e201", "e200", "e189", "e190", "e186"];
% dys_per_an = {[20:50], [7:10, 32,33,35:63,65],  [2:20, 26,27,29,31], [27:30, 32,33,34,36,38,40:75], ...
%     [62:70, 72,73,74, 76, 80:90], [7,8,10,11:15,17:21,24:42,44:46], [6:9, 11,13,15:19,21,22,24,27:29,33:35,40:43,45], ...
%     [1:51]};
anms = ["e217"];
dys_per_an = {[2 3 4 5]};
pln = 0;
% an = 'e190';%an='e189';
% individual day analysis 
% dys = [20:50]; % e218
% dys = [7:10, 32,33,35:63,65]; % e216
% dys = [2:20, 26,27]; %e217
% dys = [27:30, 32,33,34,36,38,40:75]; % e201
% dys = [62:70, 72,73,74, 76, 80:90]; % e200
% dys = [7,8,10,11:15,17:21,24:42,44:46]; % e189
% dys = [6:9, 11,13,15:19,21,22,24,27:29,33:35,40:43,45]; % e190
% dys = [1:51]; % e186
src = 'Y:\analysis\fmats';

for ii=1:length(anms)
    an = anms(ii); dys = dys_per_an{ii};    
for dy=dys % for loop per day
    clearvars -except dys an cc dy src savedst pptx anms dys_per_an ii pln
    % pth = dir(fullfile(src, an, string(dy), '**\*Fall.mat'));
    pth = dir(fullfile(src, an, 'days', sprintf('%s_day%03d*plane%i*', an, dy, pln)));
    disp(pth.name)
    % load vars
    load(fullfile(pth.folder,pth.name), 'dFF', ...
        'Fc3', 'stat', 'iscell', 'ybinned', 'changeRewLoc', ...
        'forwardvel', 'licks', 'trialnum', 'rewards', 'putative_pcs', 'VR')
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CHECKS %%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%
   
        % get suite2p indices
        pcs = reshape(cell2mat(putative_pcs), [length(putative_pcs{1}), length(putative_pcs)]);
        pc = logical(iscell(:,1))';
        [~,bordercells] = remove_border_cells_all_cells(stat, Fc3);        

        cellind = 1:size(Fc3,2);    
        pyr_tc_s2p_cellind = cellind(pc & ~bordercells); % get indices of cells that go into tuning curve
        pc_tc_s2p_cellind = pyr_tc_s2p_cellind(any(pcs,2));
    % also append fall with tables    
    save(fullfile(pth.folder,pth.name), 'pc_tc_s2p_cellind', 'pyr_tc_s2p_cellind', '-append')
end
end