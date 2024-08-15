% Zahra
% get cells detected in cellreg and do analysis

clear all; close all; clear all
src =  'Y:\analysis'; % main folder for analysis
animal = 'z8';
fld = sprintf('%s_daily_tracking_plane0',animal);
pth = dir(fullfile(src, 'celltrack', fld, "Results\*cellRegistered*"));
load(fullfile(pth.folder, pth.name))
% find cells in all sessions
[r,c] = find(cell_registered_struct.cell_to_index_map~=0);
[counts, bins] = hist(r,1:size(r,1));
sessions=length(cell_registered_struct.centroid_locations_corrected);% specify no of sessions
cindex = bins(counts>=2); % finding cells in at least 2 sessions
commoncells=zeros(length(cindex),sessions);
% make matrix of commoncells
for ci=1:length(cindex)
     commoncells(ci,:)=cell_registered_struct.cell_to_index_map(cindex(ci),:);
end
commoncells_once_per_week = commoncells;
% save
save(fullfile(pth.folder,'commoncells_once_per_week.mat'),'commoncells_once_per_week') 
fprintf('\n *************number of common cells: %i************* \n', size(commoncells,1))

%%
 % find indicies of cells active at least once a week
weekind = {};
for week=1:length(weekl)
    weekmap = cell_registered_struct.cell_to_index_map(:, weeksplit==weekl(week));
    [r,c] = find(weekmap~=0);
    [counts, bins] = hist(r,1:size(r,1));
    weekind{week} = bins(counts>=2); % finding cells in all sessions
end
lln = cellfun(@length, weekind);
% get week with most cells
weekmax = find(lln==max(lln));
otherweeks = weekl(weekl~=weekmax);
iind_activeonceweek = zeros(length(otherweeks), length(weekind{weekmax}));
for wk=1:length(otherweeks)
    active_other_weeks = ismember(weekind{weekmax},weekind{wk});
    iind_activeonceweek(wk,:) = active_other_weeks;
end
bool_activeonceweek = sum(iind_activeonceweek,1)==length(otherweeks);
iind = weekind{weekmax};
iind_activeonceweek = iind(bool_activeonceweek);
commoncells_once_per_week=zeros(length(iind_activeonceweek),sessions);
% make matrix of commoncells
for ci=1:length(iind_activeonceweek)
     commoncells_once_per_week(ci,:)=cell_registered_struct.cell_to_index_map(iind_activeonceweek(ci),:);
end
save(fullfile(pth.folder,'commoncells_once_per_week.mat'),'commoncells_once_per_week') 
fprintf('\n *************number of common cells: %i************* \n', size(commoncells,1))
fprintf('\n *************number of common cells detected once per week: %i************* \n', size(commoncells_once_per_week,1))

%%
% load mats from all days
fls = dir(fullfile(src, 'fmats', sprintf('%s\\days\\%s*.mat', animal, animal)));%dir('Z:\cellreg1month_Fmats\*YC_Fall.mat');
days = cell(1, length(fls));
for fl=1:length(fls)
    disp(fl)
    day = fls(fl);
    days{fl} = load(fullfile(day.folder,day.name), 'ops', 'stat');
end
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ANALYSIS AND PLOTS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cc=commoncells_once_per_week;
% colormap to iterate thru
ctab = hsv(length(cc));
% align all cells across all days in 1 fig
figure('Renderer','painters')
axes=zeros(1,sessions-1);
cells_to_plot = 1:size(cc,1);
for ss=1:sessions
    day=days{ss};
    axes(ss)=subplot(ceil(sqrt(sessions)), ceil(sqrt(sessions)),ss);%(4,5,ss); % 2 rows, 3 column, 1 pos; 20 days
    % figure('Renderer','painters')
    imagesc((day.ops.meanImg))
    colormap('gray')
    hold on    
    for i=cells_to_plot
        try % in case selecting cells that do not exist in every week
            x = double(day.stat{1,cc(i,ss)}.xpix');
            y = double(day.stat{1,cc(i,ss)}.ypix');
            k=boundary(x,y);
            plot(x(k),y(k), 'y');
        end
    end
    axis off    
    title(sprintf('Day %i', ss))
end
linkaxes(axes, 'xy')
