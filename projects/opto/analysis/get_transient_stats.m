function [mean_length_of_transients_per_cell_opto,....
    mean_length_of_transients_per_cell_ctrl, auc_transients_per_cell_opto,...
    auc_transients_per_cell_ctrl] = get_transient_stats(changeRewLoc, VR,ybinned, putative_pcs, Fc3, iscell,...
    stat, optoep)
% test script for counting transients
% compare to previos epoch
% zahra sees decrease in length of transients in opto but not preopto or
% other interneuron opto data...
% load('X:\vipcre\e218\35\231129_ZD_000_000\suite2p\plane0\Fall.mat')
% load("X:\vipcre\e216\9\231027_ZD_000_001\suite2p\plane0\Fall.mat")
% load("Y:\analysis\fmats\e186\days\e186_day006_plane0_Fall.mat")
if optoep<2 % take ep 2 from control days
    optoep = 2;
end
% preprocessing
eps = find(changeRewLoc>0);
eps = [eps length(changeRewLoc)];
track_length = 180/VR.scalingFACTOR;
bin_size=3;
nbins = track_length/bin_size;
ybinned = ybinned/VR.scalingFACTOR;
rewlocs = changeRewLoc(changeRewLoc>0)/VR.scalingFACTOR;
rewsize = VR.settings.rewardZone/VR.scalingFACTOR;
eprng = eps(optoep):eps(optoep+1);
ypos = ybinned(eprng);
eprng = eprng(ypos<rewlocs(optoep)); % pre reward ypos
pcs = reshape(cell2mat(putative_pcs), [length(putative_pcs{1}), length(putative_pcs)]);
pc = logical(iscell(:,1));
[~,bordercells] = remove_border_cells_all_cells(stat, Fc3);        
bordercells_pc = bordercells(pc); % mask border cells
fc3_pc = Fc3(eprng,pc); % only iscell
fc3_pc = fc3_pc(:,~bordercells_pc); % remove border cells
fc3_pc = fc3_pc(:, any(pcs,2)); % apply place cell filter, if a cell is considered a place cell in any ep!!
start_of_transients = cell(1, size(fc3_pc,2));
length_of_transients = cell(1, size(fc3_pc,2));
auc_of_transients = cell(1, size(fc3_pc,2));
for cll=1:size(fc3_pc,2) % get transients of each cell
    transient = consecutive_stretch(find(fc3_pc(:,cll)>0));
    start_of_transients{cll} = cell2mat(cellfun(@min, transient, 'UniformOutput', false));
    auc_of_transients{cll} = cell2mat(cellfun(@trapz, transient, 'UniformOutput', false));
    length_of_transients{cll} = cell2mat(cellfun(@length, transient, 'UniformOutput', false));
    clear transient
end
% get number and length
auc_transients_per_cell_opto = cell2mat(cellfun(@mean, auc_of_transients, 'UniformOutput',false));
mean_length_of_transients_per_cell_opto = cell2mat(cellfun(@(x) mean(x./31.25), length_of_transients, 'UniformOutput', false));
% previous opto
eprng = eps(optoep-1):eps(optoep);
ypos = ybinned(eprng);
eprng = eprng(ypos<rewlocs(optoep)); % pre reward ypos
fc3_pc = Fc3(eprng,pc); % only iscell
fc3_pc = fc3_pc(:,~bordercells_pc); % remove border cells
fc3_pc = fc3_pc(:, any(pcs,2)); % apply place cell filter, if a cell is considered a place cell in any ep!!
start_of_transients = cell(1, size(fc3_pc,2));
length_of_transients = cell(1, size(fc3_pc,2));
auc_of_transients = cell(1, size(fc3_pc,2));
for cll=1:size(fc3_pc,2) % get transients of each cell
    transient = consecutive_stretch(find(fc3_pc(:,cll)>0));
    start_of_transients{cll} = cell2mat(cellfun(@min, transient, 'UniformOutput', false));
    auc_of_transients{cll} = cell2mat(cellfun(@trapz, transient, 'UniformOutput', false));
    length_of_transients{cll} = cell2mat(cellfun(@length, transient, 'UniformOutput', false));
    clear transient
end
% get number and length
auc_transients_per_cell_ctrl = cell2mat(cellfun(@mean, auc_of_transients, 'UniformOutput',false));
mean_length_of_transients_per_cell_ctrl = cell2mat(cellfun(@(x) mean(x./31.25), length_of_transients, 'UniformOutput', false));
% 
% figure; plot(1,number_of_transients_per_cell_ctrl,'ko'); hold on; 
% plot(2,number_of_transients_per_cell_opto,'ro');
% [h,p,i,stats] = ttest(number_of_transients_per_cell_opto, number_of_transients_per_cell_ctrl)
% 
% figure; plot(1,mean_length_of_transients_per_cell_ctrl,'ko'); hold on; 
% plot(2,mean_length_of_transients_per_cell_opto,'ro'); xlim([0 3])
% [h,p,i,stats] = ttest(mean_length_of_transients_per_cell_opto, mean_length_of_transients_per_cell_ctrl)
end