% Zahra
% get cells detected in cellreg and do analysis

clear all
src = 'Y:\sstcre_analysis\'; % main folder for analysis
animal = 'e201';
weekfld = 'day28_31';
pth = dir(fullfile(src, "celltrack", sprintf([animal, '_', weekfld]), "Results\*cellRegistered*"));
load(fullfile(pth.folder, pth.name))
% find cells in all sessions
[r,c] = find(cell_registered_struct.cell_to_index_map~=0);
[counts, bins] = hist(r,1:size(r,1));
sessions=length(cell_registered_struct.centroid_locations_corrected);% specify no of sessions
cindex = bins(counts==sessions); % finding cells AT LEAST 2 SESSIONS???
commoncells=zeros(length(cindex),sessions);
for ci=1:length(cindex)
    commoncells(ci,:)=cell_registered_struct.cell_to_index_map(cindex(ci),:);
end
 
% load mats from all days
fls = dir(fullfile(src, 'fmats', sprintf('%s_param_test\\fall\\*.mat', animal)));%dir('Z:\cellreg1month_Fmats\*YC_Fall.mat');
days = cell(1, length(fls));
for fl=1:length(fls)
    day = fls(fl);
    days{fl} = load(fullfile(day.folder,day.name));
end
cc=commoncells;
% colormap to iterate thru
ctab = hsv(length(cc));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ANALYSIS AND PLOTS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% align all cells across all days in 1 fig
figure;
axes=zeros(1,sessions);
for ss=1:sessions
    day=days(ss);day=day{1};
    axes(ss)=subplot(1,2,ss);%(4,5,ss); % 2 rows, 3 column, 1 pos; 20 days
    imagesc(day.ops.meanImg)
    colormap('gray')
    hold on;
    for i=1:length(commoncells)
        try % in case selecting cells that do not exist in every week
            plot(day.stat{1,cc(i,ss)}.xpix, day.stat{1,cc(i,ss)}.ypix, 'Color', [ctab(i,:) 0.3]);
        end
    end
    axis off    
    title(sprintf('week %i', ss))
end
linkaxes(axes, 'xy')
%savefig(sprintf("Z:\\202300201cells.fig"))
