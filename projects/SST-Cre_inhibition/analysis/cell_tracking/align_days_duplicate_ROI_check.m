%ed's init code to deal with ROIs that can be assigned to multiple ROIs
%likely we would need to plot and manually assign these to the 'correct'
%cell
%right now you'll need to change the paths for the custom weeks 
%load week specific align.mat
%drop rows with NaNs (cells that don't map to all days)
LUT_clean=[LUT, rois]; % added week index 
LUT_clean(any(isnan(LUT_clean), 2), :) = [];

for i =1:numel(LUT_clean(:,1))%loop through all rois in column 1
    dupes(i,1)=numel(find(LUT_clean(:,1)==LUT_clean(i,1)));%find total instances in column with that roi
 end

dupIdx=find(dupes(:,1)>1);%idx of duplicates. nan=0, uniques=1, duplicates>1
dupROIs=LUT_clean(dupIdx,1);

% save cleaned LUT
save('Z:\imaging_yc\week2\suite2p\plane0\LUT_clean_wo_dup_EHmethod_noiscell.mat', 'LUT_clean')

% load mats from all days
fls = dir('Z:\week2day_mapping_cellreg\week1\*YC_Fall.mat');%dir('Z:\cellreg1month_Fmats\*YC_Fall.mat');
days = cell(1, length(fls));
for fl=1:length(fls)
    day = fls(fl);
    days{fl} = load(fullfile(day.folder,day.name));
end

%days not week
sessions_total=4;

% plot duplicate ROI mappings to compare what is best
for d=1:length(dupIdx)
    cellno=dupIdx(d); %204,220 indices that are duplicates of a cell from day 1
    ctab = hsv(length(LUT_clean));
    %multi plot of cell mask across all 5 days
    figure; 
    axes=cell(1,sessions_total);
    for ss=1:sessions_total        
        day=days(ss);day=day{1};
        axes{ss}=subplot(3,2,ss); % 2 rows, 3 column, 1 pos; 20 days
        imagesc(day.ops.meanImg) %meanImg or max_proj
        colormap('gray')
        hold on;
        try
            plot(day.stat{1,LUT_clean(cellno,ss)}.xpix, day.stat{1,LUT_clean(cellno,ss)}.ypix, 'Color', [ctab(cellno,:) 0.3]);
        end
        axis off
        title(sprintf('day %i', ss)) %sprintf('day %i', ss)
        %title(axes{ss},sprintf('Cell %0d4', i))
    end
    linkaxes([axes{:}], 'xy')
end
%remove any duplicates you don't like and save the clean LUT
%NEED TO MANUALLY CHANGE!!!
LUT_clean(dupIdx([2,4,6]),:)=[];
save('Z:\imaging_yc\week1\suite2p\plane0\LUT_clean_wo_dup_EHmethod_noiscell.mat', 'LUT_clean')
