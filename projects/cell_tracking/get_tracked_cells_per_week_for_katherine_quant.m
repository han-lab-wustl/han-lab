 % Zahra
% get cells detected in cellreg and do analysis

clear all
src = 'Y:\sstcre_analysis\celltrack\old\KEEP_TO_CORRELATE_WITH_KATHERINES_ANALYSIS'; % main folder for analysis
animal = 'e201';
weekfld = 'week12-15';
pth = dir(fullfile(src,  sprintf([animal, '_', weekfld]), "Results\*cellRegistered*"));
load(fullfile(pth.folder, pth.name))
% find cells in all sessions
[r,c] = find(cell_registered_struct.cell_to_index_map~=0);
[counts, bins] = hist(r,1:size(r,1));
sessions=length(cell_registered_struct.centroid_locations_corrected);% specify no of sessions
cindex = bins(counts>=sessions); % finding cells AT LEAST 2 SESSIONS???
commoncells=zeros(length(cindex),sessions);
for ci=1:length(cindex)
    commoncells(ci,:)=cell_registered_struct.cell_to_index_map(cindex(ci),:);
end
% save cells common across all days
save(fullfile(pth.folder,'commoncells.mat'), 'commoncells')
p_same_mat = cell_registered_struct.p_same_registered_pairs(cindex) ;
for i=1:length(p_same_mat)
    mean_Psame{i}=mean(mean(p_same_mat{i,1},'omitnan'),'omitnan');
end
mean_Psame_mat = cell2mat(mean_Psame)';
cc=commoncells;
% calculate average p-same for validation of probabilistic model
