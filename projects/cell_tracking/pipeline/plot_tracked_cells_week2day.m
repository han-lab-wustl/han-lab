function [savepth, cellmap2dayacrossweeks] = plot_tracked_cells_week2day(src, animal, ...
    weekfld, weeknms, sessions_total)
% session_total = numof days imaged.
% Zahra
% get cells detected in cellreg and do analysis
% find cells detected in all 4 weeks (transform 1)
% we want to keep all these cells
weekdst = dir(fullfile(src, sprintf([animal, '_', weekfld]), "Results\*cellRegistered*"));
weeks = load(fullfile(weekdst.folder,weekdst.name));
% find cells in all sessions
[r,c] = find(weeks.cell_registered_struct.cell_to_index_map~=0); % exclude week 1
[counts, bins] = hist(r,1:size(r,1));
sessions=length(weeks.cell_registered_struct.centroid_locations_corrected);% specify no of sessions
cindex = bins(counts==sessions); % finding cells AT LEAST 2 SESSIONS???
commoncells_4weeks=zeros(length(cindex),sessions);
for ci=1:length(cindex)
    commoncells_4weeks(ci,:)=weeks.cell_registered_struct.cell_to_index_map(cindex(ci),:);
end
% % for each of these cells, if this cell maps to day 1, or day 1,2,3, etc...
% % find those cells that map to atleast 1 day 
wkcount = 1;
for week=weeknms % for e201, excluded week 1
    week2daynm = dir(fullfile(src, sprintf([animal, '_', ...
        'week%02d_plane0_to_days'], week), ...
        "Results\*cellRegistered*"));
    week2day = load(fullfile(week2daynm.folder,week2daynm.name));
    % find cells in all sessions
    [r,c] = find(week2day.cell_registered_struct.cell_to_index_map~=0);
    [counts, bins] = hist(r,1:size(r,1));
    sessions=size(week2day.cell_registered_struct.cell_to_index_map,2);% specify no of sessions, exclude week
    cindex = bins(counts>=2); % finding cells across only 1 session + the week 
    % week has to have an index becuase it is a ref
    commoncells=zeros(length(cindex),sessions);
    for ci=1:length(cindex)
        commoncells(ci,:)=week2day.cell_registered_struct.cell_to_index_map(cindex(ci),:);
    end    
    % make sure cells map across weeks
    % uses suite2p indices
    mask = ismember(commoncells(:,sessions),commoncells_4weeks(:,wkcount));
    commoncells_mapped = commoncells(mask,:);
    save(fullfile(week2daynm.folder, 'commoncells_in_more_than_onedayofweek.mat'),'commoncells_mapped')
    wkcount=wkcount+1; % add to counter because week numbers do not always start from 1
end

weektodays = dir(fullfile(src, sprintf([animal, '_', 'week*_to_days'])));
week_maps = cell(1,length(weeknms));
wkcount = 1;
for i=weeknms
    week_map=load(fullfile(weektodays(wkcount).folder,weektodays(wkcount).name, ...
        "Results\commoncells_in_more_than_onedayofweek.mat")).commoncells_mapped;
    week_maps{wkcount}=week_map;
    wkcount = wkcount + 1;
end

% for each week, we need to find cells that 1) map to days of that week
% (already in week1,2,... arrays) and 2) map to other days of the week
% need logicals i.e cell 1 in week 1 is in week 2 and week 3 and week 4
% see below...
week1cells_to_map=commoncells_4weeks(:,1); % start with all cells across weeks
cellmap2dayacrossweeks=zeros(length(week1cells_to_map),sessions_total);
for w=1:length(week1cells_to_map)
    %cell index in other weeks
    week1cell=week1cells_to_map(w);
    cell_across_weeks=commoncells_4weeks(find(commoncells_4weeks(:,1)==week1cell),:);
    for wk=1:size(commoncells_4weeks,2)
        tweek=week_maps{wk};
        dayscell=tweek(find(tweek(:,end)==cell_across_weeks(wk)),1:end-1); % exclude last column which is 
        % the week
        if isempty(dayscell) % if that cell is not tracked that week, add 0's to all days
            daysweekcell{wk}=zeros(1,size(dayscell,2));
        else
            daysweekcell{wk}=dayscell;
        end
    end
    cellmap2dayacrossweeks(w,:) = [daysweekcell{:}];
    % this matrix is typically larger than the week2day ones!!! because it
    % includes cells that were tracked across weeks but not tracked to days
    % in one of the weeks
    % adds back those cells that i removed previously! but ok
end

%save
savepth = fullfile(weekdst.folder, ...
    "week2daymap.mat");
save(savepth, "cellmap2dayacrossweeks")
%%

% load mats from all days
fls = dir(fullfile(src, "fmats",animal, 'days', '*day*_Fall.mat'));
days = cell(1, length(fls));
for fl=1:length(fls)
    disp(fl);
    dy = fls(fl);
    days{fl} = load(fullfile(dy.folder,dy.name)) ; %'ops', 'stat', 'dFF'
end

%%
%%%%%%%%%%%%%%%%%%%%%% figures for validation %%%%%%%%%%%%%%%%%%%%%%

% align all cells across all days in 1 fig
% colormap to iterate thru
cc=cellmap2dayacrossweeks;
ctab = hsv(length(cc));
figure;
axesnm=zeros(1,sessions_total);
for ss=1:sessions_total
    day=days(ss);day=day{1};
    axesnm(ss)=subplot(ceil(sqrt(sessions_total)),ceil(sqrt(sessions_total)),ss);%(4,5,ss); % 2 rows, 3 column, 1 pos; 20 days
    imagesc(day.ops.meanImg)
    colormap('gray')
    hold on;
    for i=1:length(cc)
        try
            plot(day.stat{1,cc(i,ss)}.xpix, day.stat{1,cc(i,ss)}.ypix, 'Color', [ctab(i,:) 0.1]);
        end
    end
    axis off
    [daynm,~] = fileparts(day.ops.data_path); % relies on naming structure!
    [~,daynm] = fileparts(daynm);
%     title(sprintf('day %s', daynm))
    title(sprintf('day %i', ss))
end
linkaxes(axesnm, 'xy')

%%
% plot traces across all days
cellno=randi(length(cc));
figure;
for dayplt=1:sessions_total
    ax1=subplot(sessions_total,1,dayplt);
    day=days(dayplt);day=day{1};
    try
        plot(day.dFF(:,cc(cellno,dayplt)),'k') % 2 in the first position is cell no
    end
    axs{dayplt}=ax1;
    set(gca,'XTick',[], 'YTick', [])
    set(gca,'visible','off')
end
% linkaxes([axs{:}],'xy')
copygraphics(gcf, 'BackgroundColor', 'none');
title(sprintf('Cell no. %03d', cellno));

end