% Zahra
% get cells detected in cellreg and do analysis
clear all;
% find cells detected in all 4 weeks (transform 1)
% we want to keep all these cells
src = 'Y:\sstcre_analysis\'; % main folder for analysis
animal = 'e201';
weeknm = 1:6;
weekfld = sprintf('week%i-%i', min(weeknm), max(weeknm));
weekdst = dir(fullfile(src, "celltrack", sprintf([animal, '_', weekfld]), "Results\*cellRegistered*"));
weeks = load(fullfile(weekdst.folder,weekdst.name));
% find cells in all sessions
[r,c] = find(weeks.cell_registered_struct.cell_to_index_map(:,2:end)~=0); % exclude week 1
[counts, bins] = hist(r,1:size(r,1));
sessions=length(weeks.cell_registered_struct.centroid_locations_corrected(2:end));% specify no of sessions
cindex = bins(counts==sessions); % finding cells AT LEAST 2 SESSIONS???
commoncells_4weeks=zeros(length(cindex),sessions);
for ci=1:length(cindex)
    commoncells_4weeks(ci,:)=weeks.cell_registered_struct.cell_to_index_map(cindex(ci),2:end);
end

% for each of these cells, if this cell maps to day 1, or day 1,2,3, etc...
% find those cells that map to atleast 1 day 
for week=2:6 % for e201, excluded week 1
    week2daynm = dir(fullfile(src, "celltrack", sprintf([animal, '_', 'week%i_to_days'], week), "Results\*cellRegistered*"));
    week2day = load(fullfile(week2daynm.folder,week2daynm.name));
    % find cells in all sessions
    [r,c] = find(week2day.cell_registered_struct.cell_to_index_map~=0);
    [counts, bins] = hist(r,1:size(r,1));
    sessions=size(week2day.cell_registered_struct.cell_to_index_map,2);% specify no of sessions, exclude week
    cindex = bins(counts>=2); % finding cells across only 1 session + the week 
    commoncells=zeros(length(cindex),sessions);
    for ci=1:length(cindex)
        commoncells(ci,:)=week2day.cell_registered_struct.cell_to_index_map(cindex(ci),:);
    end    
    % make sure cells map across weeks
    mask = ismember(commoncells(:,sessions),commoncells_4weeks(:,week-1));
    commoncells_mapped = commoncells(mask,:);
    save(fullfile(week2daynm.folder, 'commoncells_in_more_than_onedayofweek_mapped_across_weeks.mat'),'commoncells_mapped')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ANALYSIS AND PLOTS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
weektodays = dir(fullfile(src, "celltrack", sprintf([animal, '_', 'week*_to_days'])));
week_maps = cell(1,max(weeknm));
for i=2:6
    week_map=load(fullfile(weektodays(i).folder,weektodays(i).name, ...
        "Results\commoncells_in_more_than_onedayofweek_mapped_across_weeks.mat")).commoncells_mapped;
    week_maps{i}=week_map;
end

% for each week, we need to find cells that 1) map to days of that week
% (already in week1,2,... arrays) and 2) map to other days of the week
% need logicals i.e cell 1 in week 1 is in week 2 and week 3 and week 4
% see below...
week1cells_to_map=commoncells_4weeks(:,1); % start with all cells across weeks
sessions_total=20; %total number of days imaged (e.g. included in dataset)
cellmap2dayacrossweeks=zeros(length(week1cells_to_map),sessions_total);
for w=1:length(week1cells_to_map)
    %cell index in other weeks
    week1cell=week1cells_to_map(w);
    cell_across_weeks=commoncells_4weeks(find(commoncells_4weeks(:,1)==week1cell),:);
    for wk=1:size(commoncells_4weeks,2)
        tweek=week_maps{wk+1}; %skipped week 1
        dayscell=tweek(find(tweek(:,end)==cell_across_weeks(wk)),1:end-1); % exclude last column which is 
        % the week
        if isempty(dayscell)
            daysweekcell{wk}=zeros(1,size(dayscell,2));
        else
            daysweekcell{wk}=dayscell;
        end
    end
    cellmap2dayacrossweeks(w,:) = [daysweekcell{:}];
end

%save
save(fullfile(weekdst.folder, ...
    "commoncells_atleastoneactivedayperweek_4weeks_week2daymap.mat"), "cellmap2dayacrossweeks")
%%
cc=cellmap2dayacrossweeks;
ctab = hsv(length(cc));

% load mats from all days
fls = dir(fullfile(src, "fmats",animal, 'days\*day*_Fall.mat'));%dir('Z:\cellreg1month_Fmats\*YC_Fall.mat');
days = cell(1, length(fls));
for fl=1:length(fls)
    day = fls(fl);
    days{fl} = load(fullfile(day.folder,day.name));
end


% figures for validation
% align each common cells across all days with an individual mask
% remember this is the cell index, so you have to find the cell in the
% original F mat

% for i=100:150
%     %multi plot of cell mask across all 5 days
%     figure(i); 
%     axes=cell(1,sessions_total);
%     for ss=1:sessions_total        
%         day=days(ss);day=day{1};
%         axes{ss}=subplot(4,5,ss); % 2 rows, 3 column, 1 pos; 20 days
%         imagesc(day.ops.meanImg) %meanImg or max_proj
%         colormap('gray')
%         hold on;
%         try
%             plot(day.stat{1,cc(i,ss)}.xpix, day.stat{1,cc(i,ss)}.ypix, 'Color', [ctab(i,:) 0.3]);
%         end
%         axis off
%         title(sprintf('day %i', ss)) %sprintf('day %i', ss)
%         %title(axes{ss},sprintf('Cell %0d4', i))
%     end
%     linkaxes([axes{:}], 'xy')
%     %savefig(sprintf("Z:\\suite2pconcat1month_commoncellmasks\\cell_%03d.fig",i+250)) %changed to reflect subset of cells plotted
% end

%%
% align all cells across all days in 1 fig
% colormap to iterate thru
ctab = hsv(length(cc));
figure;
axesnm=zeros(1,sessions_total);
for ss=1:sessions_total
    day=days(ss);day=day{1};
    axesnm(ss)=subplot(4,5,ss);%(4,5,ss); % 2 rows, 3 column, 1 pos; 20 days
    imagesc(day.ops.meanImg)
    colormap('gray')
    hold on;
    for i=1:length(cc)
        try
            plot(day.stat{1,cc(i,ss)}.xpix, day.stat{1,cc(i,ss)}.ypix, 'Color', [ctab(i,:) 0.3]);
        end
    end
    axis off
    [daynm,~] = fileparts(day.ops.data_path);
    [~,daynm] = fileparts(daynm);
    title(sprintf('day %s', daynm))
end
linkaxes(axesnm, 'xy')
%%
%calculate dff
dff={};
for i=1:length(days)
    day=days(i);day=day{1};
    dff{i}=redo_dFF(day.F, 31.25, 20, day.Fneu);
    disp(i)
end
save(fullfile(weekdst.folder, "dff_per_day.mat"), "dff", "-v7.3")
% dff=load(fullfile(weekdst.folder, "dff_per_day.mat"), "dff");
%%
% plot F (and ideally dff) over ypos

days_to_plot=[18,19,20]; %plot 5 days at a time
cellno=randi([1 500],1,1);
grayColor = [.7 .7 .7];
sessions_total=3;
fig=figure;
subplot_j=1;
for dayplt=days_to_plot
    ax1=subplot(3,1,subplot_j);
    day=days(dayplt);day=day{1};
    plot(day.ybinned, 'Color', grayColor); hold on; 
    plot(day.changeRewLoc, 'b')
    plot(find(day.licks),day.ybinned(find(day.licks)),'r.')
    yyaxis right
    try
        plot(dff{dayplt}(cc(cellno,dayplt),:),'g') % 2 in the first position is cell no
    end
    title(sprintf('day %i', dayplt))
    axs{dayplt}=ax1;
    subplot_j=subplot_j+1; % for subplot index
end
han=axes(fig,'visible','off'); 
han.Title.Visible='on';
han.XLabel.Visible='off';
han.XAxis.Visible='off';
han.YLabel.Visible='on';
ylabel(han,'Y position');
xlabel(han,'Frames');
title(han,sprintf('Cell no. %03d', cellno));

% savefig(sprintf('Z:\\cellregtest_behavior\\cell_%05d_days%02d_%02d_%02d_%02d_%02d.fig', cellno, days_to_plot))
%%
% plot traces across all days

% days_to_plot=[1,6,10,14,18]; %plot 5 days at a time
% cellno=78;
% for cellno=100:150 %no. of common cells
% grayColor = [.7 .7 .7];
% fig=figure;
% for dayplt=1:sessions_total
%     ax1=subplot(sessions_total,1,dayplt);
%     day=days(dayplt);day=day{1};
%     try
%         plot(dff{dayplt}(cc(cellno,dayplt),:),'k') % 2 in the first position is cell no
%     end
%     axs{dayplt}=ax1;
%     set(gca,'XTick',[], 'YTick', [])
%     set(gca,'visible','off')
% end
% % linkaxes([axs{:}],'xy')
% copygraphics(gcf, 'BackgroundColor', 'none');
% title(sprintf('Cell no. %03d', cellno));

%%

% convert to 1 (temp)'s to bool for reward analysis
cc(cc==0)=1;
% align to behavior (rewards and solenoid) for each cell?
% per day, get this data...
range=5;
bin=0.2;
%only get days with rewards (exclude training)
daysrewards = [9    10    11    12    13];
ccbinnedPerireward=cell(1,length(daysrewards));
ccrewdFF=cell(1,length(daysrewards));
for d=daysrewards
    day=days(d);day=day{1};
    rewardsonly=day.rewards==1;
    cs=day.rewards==0.5;
    % runs for all cells
    [binnedPerireward,allbins,rewdFF] = perirewardbinnedactivity(dff{d}', ...
        rewardsonly,day.timedFF,range,bin); %rewardsonly if mapping to reward
    % now extract ids only of the common cells
    ccbinnedPerireward{d}=binnedPerireward;
    ccrewdFF{d}=rewdFF;
end
%%
% plot
% if cell is missing from 1 day, take mean dff of others days from that
% cell??? NOT implemented yet
% optodays=[5,6,7,9,10,11,13,14,16,17,18];
cells_to_plot = 10;
for cellno=randi([1 500],1,cells_to_plot ) %random number of cells
    dd=1; %for legend
    figure;
    clear legg;
    for d=daysrewards
        pltrew=ccbinnedPerireward{d}; %temp hack that excludes cell #1
        try %if cell exists on that day, otherwise day is dropped...
%             if ~any(optodays(:)==d)            
                plot(pltrew(cc(cellno,d),:)')            
%             else
%               plot(pltrew(cc(cellno,d),:)', 'Color', 'red')    
%             end
            legg{dd}=sprintf('day %d',d); dd=dd+1;           
        end
        hold on;        
    end    
     % plot reward location as line
    xticks([1:5:50, 50])
    x1=xline(median([1:5:50, 50]),'-.b','Reward'); %{'Conditioned', 'stimulus'}
    xticklabels([allbins(1:5:end) range]);
    xlabel('seconds')
    ylabel('dF/F')
    legend(char(legg))
    title(sprintf('Cell no. %04d', cellno))
end