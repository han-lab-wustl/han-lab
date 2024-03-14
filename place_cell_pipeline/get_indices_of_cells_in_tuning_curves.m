% Zahra - March 2024
% get indices of cells that go into the tuning curves
% just needed for 
clear all; 
% individual day analysis
anms = ["e218", "e216", "e217", "e201", "e200", "e189", "e190", "e186"];
dys_per_an = {[20:50], [7:10, 32,33,35:63,65],  [2:20, 26,27], [27:30, 32,33,34,36,38,40:75], ...
    [62:70, 72,73,74, 76, 80:90], [7,8,10,11:15,17:21,24:42,44:46], [6:9, 11,13,15:19,21,22,24,27:29,33:35,40:43,45], ...
    [1:51]};
% an = 'e218'; dys = [20:50]; % e218
% an = 'e216'; dys = [7:10, 32,33,35:63,65]; % e216
% dys = [2:20, 26,27]; %e217
% dys = [27:30, 32,33,34,36,38,40:75]; % e201
% dys = [62:70, 72,73,74, 76, 80:90]; % e200
% dys = [7,8,10,11:15,17:21,24:42,44:46]; % e189
% an = 'e190'; dys = [6:9, 11,13,15:19,21,22,24,27:29,33:35,40:45]; % e190
% dys = [1:51]; % e186
for ii=1:length(anms)
    an = anms(ii); dys = dys_per_an{ii};
    src = 'Y:\analysis\fmats';
    for dy=dys % for loop per day
        clearvars -except an dys dy src
        pth = dir(fullfile(src, an, 'days', sprintf('%s_day%03d*plane0*', an, dy)));
        % load vars
        load(fullfile(pth.folder,pth.name), 'Fc3', 'stat', 'iscell')
        cellind = 1:size(Fc3,2);    
        pc = logical(iscell(:,1))';
        [~,bordercells] = remove_border_cells_all_cells(stat, Fc3);        
        pyr_tc_s2p_cellind = cellind(pc & ~bordercells);       
        save(fullfile(pth.folder,pth.name), 'pyr_tc_s2p_cellind', '-append')
        disp(fullfile(pth.folder,pth.name))
    end
end
