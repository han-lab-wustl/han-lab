% zahra
% get activity of tracked cells per day
clear all;
load("Y:\analysis\celltrack\e218_week01-06_plane0\Results\commoncells_atleastoneactivedayperweek_4weeks_week2daymap.mat",...
    "cellmap2dayacrossweeks")
src = "Y:\analysis\fmats";
mouse_name = 'e218';
cc=cellmap2dayacrossweeks;

dys = [20,21,22,23,35,36,37,38,39,40,41,...
     42,43,44,45,47 48 49 50];
tracked_dys = [1 2 3 4 16:26 28:31]; % corresponding to days analysing

opto_ep = [-1 -1 -1 -1 3 0 1 2 0 1 3 0 1 2 0 3 0 1 2];
for dy=1:length(dys)
    clearvars -except dy src mouse_name dys cc opto_ep tracked_dys
    daypth = dir(fullfile(src, mouse_name, 'days', sprintf('%s_day%03d*.mat', mouse_name, dys(dy)))); 
    load(fullfile(daypth.folder,daypth.name));
    pcs = reshape(cell2mat(putative_pcs), [length(putative_pcs{1}), length(putative_pcs)]);
    tracked_cells = cc(:,tracked_dys(dy));    
    % get activity on opto ep
    optoep = opto_ep(dy);
    if optoep<2
        optoep=2;
    end
    try
        gainf = 1/VR.scalingFACTOR;
    catch
        gainf = 3/2; % 3/2 VS. 1; in this pipeline the gain is multiplied everywhere
    end
    ybinned = ybinned*gainf; rewlocs = changeRewLoc(changeRewLoc>0)*(gainf); 
    eps = find(changeRewLoc>0);
    eps = [eps length(changeRewLoc)];      
    eprng = eps(optoep):eps(optoep+1); 
    eprng = eprng(ybinned(eprng)<rewlocs(optoep)); % pre reward only
    tracked_cell_iind = 1:length(tracked_cells);
    tracked_cells_this_day_iind = tracked_cell_iind(tracked_cells>0); % important var to save
    dff_tracked_cells = dFF(eprng,  tracked_cells(tracked_cells>0));
    fc3_tracked_cells = Fc3(:,  tracked_cells(tracked_cells>0));
    mean_dff_per_cell_opto = mean(dff_tracked_cells,1,'omitnan');
     [mean_length_of_transients_per_cell_opto,....
    mean_length_of_transients_per_cell_ctrl, auc_transients_per_cell_opto,...
    auc_transients_per_cell_ctrl, peak_transients_per_cell_opto, ...
    peak_transients_per_cell_ctrl] = get_transient_stats(changeRewLoc, gainf, ybinned, putative_pcs, fc3_tracked_cells, iscell,...
    stat, optoep);
    % previous ep
    optoep = optoep-1;
    eprng = eps(optoep):eps(optoep+1);    
    eprng = eprng(ybinned(eprng)<rewlocs(optoep));
    tracked_cell_iind = 1:length(tracked_cells);
    tracked_cells_this_day_iind = tracked_cell_iind(tracked_cells>0); % important var to save
    dff_tracked_cells = dFF(eprng,  tracked_cells(tracked_cells>0));
    mean_dff_per_cell_prev = mean(dff_tracked_cells,1,'omitnan');
    tracked_cells_dff_fc3(1,:) = tracked_cells_this_day_iind;
    tracked_cells_dff_fc3(2,:) = mean_dff_per_cell_prev;
    tracked_cells_dff_fc3(3,:) = mean_dff_per_cell_opto;       
    tracked_cells_dff_fc3(4,:) = auc_transients_per_cell_ctrl;    
    tracked_cells_dff_fc3(5,:) = auc_transients_per_cell_opto;    

    tracked_cells_dff_fc3 = tracked_cells_dff_fc3';
    tracked_cells_dff_fc3_table = array2table(tracked_cells_dff_fc3, ...
        'VariableNames', {'tracked_cell_index', 'mean_dff_per_cell_prev', 'mean_dff_per_cell_opto', 'auc_transients_per_cell_ctrl', ...
        'auc_transients_per_cell_opto'});
    save(fullfile(daypth.folder,daypth.name), 'tracked_cells_dff_fc3_table', '-append')
    disp(fullfile(daypth.folder,daypth.name))
    % pciind = 1:size(dFF,2);
    % pc = logical(iscell(:,1));
    % pciind = pciind(pc);
    % bordercells_pc = bordercells(pc); % mask border cells
    % pciind = pciind(~bordercells_pc);
    % tracked_cells_this_day_iind_pcs = pciind(ismember(pciind,tracked_cells_this_day_iind)); % place cell indicies
end
%%
% compile tracked cell activity across days
tracked_cells_dff_fc3_tables = {};
for dy=1:length(dys)
    daypth = dir(fullfile(src, mouse_name, 'days', sprintf('%s_day%03d*.mat', mouse_name, dys(dy)))); 
    load(fullfile(daypth.folder,daypth.name), 'tracked_cells_dff_fc3_table');
    tracked_cells_dff_fc3_tables{dy} = tracked_cells_dff_fc3_table;
    disp(fullfile(daypth.folder,daypth.name));
end
%%
% plot each cell in prev vs. opto ep
opto_tracked = tracked_cells_dff_fc3_tables(opto_ep>1);
tracked_iind = opto_tracked{1}.tracked_cell_index; 
mat = {}; idx = 1;
for iind=1:length(tracked_iind)
try
opto_mean_auc_across_sessions = cellfun(@(x) x(x.tracked_cell_index==tracked_iind(iind),'auc_transients_per_cell_opto'), opto_tracked, 'UniformOutput',false);
% fix format
opto_mean_auc_across_sessions =cell2mat(cellfun(@(x) x.auc_transients_per_cell_opto, opto_mean_auc_across_sessions, 'UniformOutput',false));

ctrl_mean_auc_across_sessions = cellfun(@(x) x(x.tracked_cell_index==tracked_iind(iind),'auc_transients_per_cell_ctrl'), opto_tracked, 'UniformOutput',false);
% fix format
ctrl_mean_auc_across_sessions = cell2mat(cellfun(@(x) x.auc_transients_per_cell_ctrl, ctrl_mean_auc_across_sessions, 'UniformOutput',false));
mat{idx,1} = ctrl_mean_auc_across_sessions;
mat{idx,2} = opto_mean_auc_across_sessions;
mat{idx,3} = ctrl_mean_auc_across_sessions-opto_mean_auc_across_sessions;
idx = idx+1;
catch
    fprintf('cell %i likely not tracked all days \n', iind)
end
end
%%
mat_ = cell2mat(mat);
diffmat = mat_(:,end-5:end);
diffmat_mean = mean(diffmat,2,'omitnan');
[~,sortidx] = sort(diffmat_mean, 'descend');
diffmat_mean = diffmat_mean(sortidx,:);
mat_sort = mat_(sortidx,:);
figure; imagesc(normalize(mat_sort(:,1:end-6)))
%%
iind_to_test = 1:length(diffmat_mean);
iind_to_test = iind_to_test(sum(diffmat>0,2)>4); % ctrl is higher
for ii=iind_to_test%size(mat_,1)
    prev = mat_(ii,1:ll); opto = mat_(ii,ll+1:ll+ll);
    figure; plot(1,prev, 'ko'); hold on; plot(2,opto, 'ro');
    xlim([0 3])
    for j=1:length(mat_(ii,1:ll))
        plot([1 2], [prev(j), opto(j)],'k--')
    end
    title(sprintf('Tracked cell no %i', ii))
end
% yticks(1:length(sortidx)); yticklabels(sortidx)