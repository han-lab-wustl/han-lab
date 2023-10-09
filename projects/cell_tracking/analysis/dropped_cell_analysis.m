% goal is to label cells that are dropped on one day by
% 1) blue if there are cells around it
% 2) red if there are no cells within a 100 pix radius
clear all
weeknm = 13;
wknml = 2;  % which number in seq
fls = dir(fullfile('Y:\sstcre_analysis\fmats\e201', sprintf('week%i', weeknm),'day*_Fall.mat'));%dir('Z:\cellreg1month_Fmats\*YC_Fall.mat');
days = cell(1, length(fls));
for fl=1:length(fls)
    disp(fl);
    dy = fls(fl);
    days{fl} = load(fullfile(dy.folder,dy.name), 'stat', 'ops', 'cc');
end
cc=days{1}.cc;
sessions_total=length(days);
wkfl = dir(fullfile('Y:\sstcre_analysis\fmats\e201', sprintf('week%i', weeknm), 'week*_Fall.mat'));%dir('Z:\cellreg1month_Fmats\*YC_Fall.mat');
wk = load(fullfile(wkfl.folder,wkfl.name), 'stat', 'ops');
weeks=load('Y:\sstcre_analysis\celltrack\e201_week12-15\Results\cellRegistered_20230815_192855.mat');
[r,c] = find(weeks.cell_registered_struct.cell_to_index_map~=0); % exclude week 1
[counts, bins] = hist(r,1:size(r,1));
sessions=length(weeks.cell_registered_struct.centroid_locations_corrected);% specify no of sessions
cindex = bins(counts==sessions); % finding cells AT LEAST 2 SESSIONS???
commoncells_4weeks=zeros(length(cindex),sessions);
for ci=1:length(cindex)
    commoncells_4weeks(ci,:)=weeks.cell_registered_struct.cell_to_index_map(cindex(ci),:);
end
%%
figure;
r = 50;
axesnm=zeros(1,sessions_total);
for ss=1:sessions_total
    day=days(ss);day=day{1};    
    r = 50; % radius to look around cell
    missing_cell_ind = find(cc(:,ss)==0);
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
                if sum(ismember(yunit, day.stat{1,cc(j,ss)}.ypix))>0 && sum(ismember(xunit, ...
                        day.stat{1,cc(j,ss)}.xpix))>0 % checks if there are cells that day around it
                    xybool(j) = 1;
                end
            end
        end
%         plot(wk.stat{1,missing_cell_in_week(ii)}.xpix, wk.stat{1,missing_cell_in_week(ii)}.ypix, 'g');
        if sum(xybool)>20% if there are 5 cells around it
            plot(wk.stat{1,missing_cell_in_week(ii)}.xpix, wk.stat{1,missing_cell_in_week(ii)}.ypix, 'y');
        else
            plot(wk.stat{1,missing_cell_in_week(ii)}.xpix, wk.stat{1,missing_cell_in_week(ii)}.ypix, 'r');
        end
    end

    axis off
%     [daynm,~] = fileparts(day.ops.data_path);
%     [~,daynm] = fileparts(daynm);
%     title(sprintf('day %s', daynm))
end
linkaxes(axesnm, 'xy')