 % Zahra
% get cells detected in cellreg and do analysis

clear all
src = 'Y:\sstcre_analysis\'; % main folder for analysis
animal = 'e200';
weekfld = 'week09-12_plane0';
pth = dir(fullfile(src, "celltrack", sprintf([animal, '_', weekfld]), "Results\*cellRegistered*"));
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

% load mats from all weeks
fls = dir(fullfile(src, 'fmats', animal, sprintf('%s_week*_plane0*.mat',animal))); %dir(fullfile(src, 'fmats', sprintf('%s\\%s_week*.mat', animal,animal)));
days = cell(1, length(fls));
for fl=1:length(fls)
    day = fls(fl);
    days{fl} = load(fullfile(day.folder,day.name),'stat', 'ops');
end
% colormap to iterate thru
ctab = hsv(length(cc));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ANALYSIS AND PLOTS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% align all cells across all days in 1 fig
figure;
axes=zeros(1,sessions);
cells_to_plot = [1:size(cc,1)];
for ss=1:sessions
    day=days(ss);day=day{1};
    axes(ss)=subplot(2,2,ss);%(4,5,ss); % 2 rows, 3 column, 1 pos; 20 days
    imagesc(day.ops.meanImg)
    colormap('gray')
    hold on;
    for i=cells_to_plot%length(commoncells)
        try % in case selecting cells that do not exist in every week
            plot(day.stat{1,cc(i,ss)}.xpix, day.stat{1,cc(i,ss)}.ypix, 'Color', ...
                [ctab(i,:) 0.3]);
        end
    end
    axis off    
    title(sprintf('week %i', ss))
end
linkaxes(axes, 'xy')
%savefig(sprintf("Z:\\202300201cells.fig"))
