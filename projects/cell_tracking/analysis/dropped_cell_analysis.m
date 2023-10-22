% goal is to label cells that are dropped on one day by
% 1) blue if there are cells around it
% 2) red if there are no cells within a 100 pix radius
clear all
weeknm = 12;
wknml = 1;  % which number in seq
fls = dir(fullfile('Y:\sstcre_analysis\fmats\e201', ...
    'days','*day*_Fall.mat'));%dir('Z:\cellreg1month_Fmats\*YC_Fall.mat');
days = cell(1, length(fls));
for fl=1:length(fls)
    disp(fl);
    dy = fls(fl);
    days{fl} = load(fullfile(dy.folder,dy.name), 'stat', 'ops');
end
% cc=days{1}.cc;
%%
wkfl = dir(fullfile('Y:\sstcre_analysis\fmats\e201', sprintf('week%02d_plane0', weeknm), 'e201_week*_Fall.mat'));%dir('Z:\cellreg1month_Fmats\*YC_Fall.mat');
wk = load(fullfile(wkfl.folder,wkfl.name), 'stat', 'ops');
% IMPORTANT
daynm_of_total_tracked = [1 2 3 4 5]; % out of total days
days = days(daynm_of_total_tracked);
save(fullfile(wkfl.folder,wkfl.name), 'daynm_of_total_tracked', '-append')
sessions_total=length(days);

weeks=load('Y:\sstcre_analysis\celltrack\e201_week12-15_plane0\Results\cellRegistered_20231017_155101.mat');
cc = load('Y:\sstcre_analysis\celltrack\e201_week12-15_plane0\Results\commoncells_atleastoneactivedayperweek_4weeks_week2daymap.mat');
cc = cc.cellmap2dayacrossweeks;
[r,c] = find(weeks.cell_registered_struct.cell_to_index_map~=0); % exclude week 1
[counts, bins] = hist(r,1:size(r,1));
sessions=length(weeks.cell_registered_struct.centroid_locations_corrected);% specify no of sessions
cindex = bins(counts==sessions); % finding cells AT LEAST 2 SESSIONS???
commoncells_4weeks=zeros(length(cindex),sessions);
for ci=1:length(cindex)
    commoncells_4weeks(ci,:)=weeks.cell_registered_struct.cell_to_index_map(cindex(ci),:);
end
%%
% figure;
% r = 50; % radius to look around cell
% num_cells = 20; % if there are x cells around it
axesnm=zeros(1,sessions_total);
for ss=1:sessions_total
    ss_tr = daynm_of_total_tracked(ss);
    day=days(ss);day=day{1};    
    missing_cell_ind = find(cc(:,ss_tr)==0);
    missing_cell_in_week = commoncells_4weeks(missing_cell_ind,wknml);
    axesnm(ss)=subplot(3,2,ss);%(4,5,ss); % 2 rows, 3 column, 1 pos; 20 days
      
    for ii=1:length(missing_cell_ind)
        figure; 
        imagesc(day.ops.meanImg)
        colormap('gray')
        hold on;  
        x = mean(wk.stat{1,missing_cell_in_week(ii)}.xpix);
        y = mean(wk.stat{1,missing_cell_in_week(ii)}.ypix);
%         
%         th = 0:pi/50:2*pi;
%         xunit = ceil(r * cos(th) + mean(x));
%         yunit = ceil(r * sin(th) + mean(y));
%         xybool = zeros(1,length(cc));        
        dst = ones(1,length(missing_cell_ind))*NaN;
        for j=1:length(missing_cell_ind)
            try
                cy = mean(wk.stat{1,missing_cell_in_week(j)}.ypix);
                cx = mean(wk.stat{1,missing_cell_in_week(j)}.xpix);
                dst(j) = pdist([x,y; cx,cy], 'euclidean');
%                         day.stat{1,cc(j,ss_tr)}.xpix))>0 % checks if there are cells that day around it
%                     xybool(j) = 1;
            end            
        end
        dst2 = dst(~isnan(dst));
        autumn_ = vals2colormap(dst2, 'autumn');        
        for jj=1:length(missing_cell_ind) 
            plot(wk.stat{1,missing_cell_in_week(jj)}.xpix, ...
                wk.stat{1,missing_cell_in_week(jj)}.ypix, 'Color', ...
                [autumn_(jj,:) 0.3]);
        end
        plot(wk.stat{1,missing_cell_in_week(ii)}.xpix, ...
                wk.stat{1,missing_cell_in_week(ii)}.ypix, 'g');
        axis off
        title(sprintf('cell no. in week %i, day %i', missing_cell_in_week(ii), ss))
    end

    
%     [daynm,~] = fileparts(day.ops.data_path);
%     [~,daynm] = fileparts(daynm);
%     title(sprintf('day %s', daynm))
    
end

%%

figure;
r = 50; % radius to look around cell
num_cells = 20; % if there are x cells around it

axesnm=zeros(1,sessions_total);
for ss=1:sessions_total
    ss_tr = wk.daynm_of_total_tracked(ss);
    ctab = autumn(length(sst_tr));
    day=days(ss);day=day{1};    
    missing_cell_ind = find(cc(:,ss_tr)==0);
    missing_cell_in_week = commoncells_4weeks(missing_cell_ind,wknml);
    axesnm(ss)=subplot(3,2,ss);%(4,5,ss); % 2 rows, 3 column, 1 pos; 20 days
    imagesc(day.ops.meanImg)
    colormap('gray')
    hold on;    
    for ii=1:length(missing_cell_ind)
        x = wk.stat{1,missing_cell_in_week(ii)}.xpix;
        y = wk.stat{1,missing_cell_in_week(ii)}.ypix;
        th = 0:pi/50:2*pi;
        xunit = ceil(r * cos(th) + mean(x));
        yunit = ceil(r * sin(th) + mean(y));
        xybool = zeros(1,length(cc));        
        for j=1:length(cc)
            try
                if sum(ismember(yunit, day.stat{1,cc(j,ss_tr)}.ypix))>0 && sum(ismember(xunit, ...
                        day.stat{1,cc(j,ss_tr)}.xpix))>0 % checks if there are cells that day around it
                    xybool(j) = 1;
                end
            end
        end
        if sum(xybool)>num_cells 
            plot(wk.stat{1,missing_cell_in_week(ii)}.xpix, ...
                wk.stat{1,missing_cell_in_week(ii)}.ypix, 'y');
        else
            plot(wk.stat{1,missing_cell_in_week(ii)}.xpix, ...
                wk.stat{1,missing_cell_in_week(ii)}.ypix, 'r');
        end
    end

    axis off
%     [daynm,~] = fileparts(day.ops.data_path);
%     [~,daynm] = fileparts(daynm);
%     title(sprintf('day %s', daynm))
    title(sprintf('day %i', ss))
end
linkaxes(axesnm, 'xy')