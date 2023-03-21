% Zahra
% get cells detected in cellreg and do analysis

clear all
load('C:\Users\Han\Documents\MATLAB\CellReg\ZD_week2daymap_week1_20230203\Results\cellRegistered_20230203_162322.mat')
% find cells in all sessions
[r,c] = find(cell_registered_struct.cell_to_index_map~=0);
[counts, bins] = hist(r,1:size(r,1));
sessions=5;% specify no of sessions
cindex = bins(counts==sessions); % finding cells across all 5 sessions
commoncells=zeros(length(cindex),sessions);
for ci=1:length(cindex)
    commoncells(ci,:)=cell_registered_struct.cell_to_index_map(cindex(ci),:);
end
 
% load mats from all days
fls = dir('Z:\week2day_mapping_cellreg\week1\*YC_Fall.mat');%dir('Z:\cellreg1month_Fmats\*YC_Fall.mat');
days = cell(1, length(fls));
for fl=1:length(fls)
    day = fls(fl);
    days{fl} = load(fullfile(day.folder,day.name));
end
% load('Z:\\dff_221206-30.mat')
dff = cell(1,sessions);
% calculate dff across all
% for i=1:length(fls)
%     dff{i} = redo_dFF(days{i}.F, 31.25, 20, days{i}.Fneu);
%     disp(i)
% end
% save('Z:\\dff_221206-30_per_week_concat.mat', 'dff', '-v7.3')
% 
%%
% align each common cells across all days with an individual mask
ctab = hsv(length(commoncells));
% remember this is the cell index, so you have to find the cell in the
% original F mat
cc=commoncells(150:end,:);
for i=1:length(cc)
    %multi plot of cell mask across all 5 days
    figure(i); 
    axes=cell(1,sessions);
    for ss=1:sessions        
        day=days(ss);day=day{1};
        axes{ss}=subplot(3,2,ss); % 2 rows, 3 column, 1 pos; 20 days
        imagesc(day.ops.meanImg) %meanImg or max_proj
        colormap('gray')
        hold on;
        try
            plot(day.stat{1,commoncells(i,ss)}.xpix, day.stat{1,commoncells(i,ss)}.ypix, 'Color', [ctab(i,:) 0.3]);
        end
        axis off
        title(sprintf('week %i', ss)) %sprintf('day %i', ss)
    end
    linkaxes([axes{:}], 'xy')
    %savefig(sprintf("Z:\\suite2pconcat1month_commoncellmasks\\cell_%03d.fig",i+250)) %changed to reflect subset of cells plotted
end

%%
% align all cells across all days in 1 fig
% colormap to iterate thru
ctab = hsv(length(commoncells));
figure;
axes=zeros(1,sessions);
for ss=1:sessions
    day=days(ss);day=day{1};
    axes(ss)=subplot(3,2,ss);%(4,5,ss); % 2 rows, 3 column, 1 pos; 20 days
    imagesc(day.ops.meanImg)
    colormap('gray')
    hold on;
    for i=1:length(commoncells)
        try
            plot(day.stat{1,commoncells(i,ss)}.xpix, day.stat{1,commoncells(i,ss)}.ypix, 'Color', [ctab(i,:) 0.3]);
        end
    end
    axis off
    title(sprintf('day %i', ss))
end
linkaxes(axes, 'xy')
%savefig(sprintf("Z:\\202300201cells.fig"))

%%
% plot F (and ideally dff) over ypos

days_to_plot=[3,6,9,12,18]; %plot 5 days at a time
cellno=49;
grayColor = [.7 .7 .7];
fig=figure;
ax1=subplot(5,1,1);
day=days(days_to_plot(1));day=day{1};
plot(day.ybinned, 'Color', grayColor); hold on; 
plot(day.changeRewLoc, 'b')
plot(find(day.licks),day.ybinned(find(day.licks)),'r.')
yyaxis right
try
    plot(dff{days_to_plot(1)}(commoncells(cellno,days_to_plot(1)),:),'g') % 2 in the first position is cell no
end
title(sprintf('day %i', days_to_plot(1)))

ax2=subplot(5,1,2);
day=days(days_to_plot(2));day=day{1};
plot(day.ybinned, 'Color', grayColor); hold on; 
plot(day.changeRewLoc, 'b')
plot(find(day.licks),day.ybinned(find(day.licks)),'r.')
yyaxis right
try
    plot(dff{days_to_plot(2)}(commoncells(cellno,days_to_plot(2)),:),'g') % 2 in the second position here is session 2 (day 7)
end
title(sprintf('day %i', days_to_plot(2)))

ax3=subplot(5,1,3);
day=days(days_to_plot(3));day=day{1};
plot(day.ybinned, 'Color', grayColor); hold on; 
plot(day.changeRewLoc, 'b')
plot(find(day.licks),day.ybinned(find(day.licks)),'r.')
yyaxis right
try
    plot(dff{days_to_plot(3)}(commoncells(cellno,days_to_plot(3)),:),'g') 
end
title(sprintf('day %i', days_to_plot(3)))

ax4=subplot(5,1,4);
day=days(days_to_plot(4));day=day{1};
plot(day.ybinned, 'Color', grayColor); hold on; 
plot(day.changeRewLoc, 'b')
plot(find(day.licks),day.ybinned(find(day.licks)),'r.')
yyaxis right
try
    plot(dff{days_to_plot(4)}(commoncells(cellno,days_to_plot(4)),:),'g') 
end
title(sprintf('day %i', days_to_plot(4)))

ax5=subplot(5,1,5);
day=days(days_to_plot(5));day=day{1};
plot(day.ybinned, 'Color', grayColor); hold on; 
plot(day.changeRewLoc, 'b')
plot(find(day.licks),day.ybinned(find(day.licks)),'r.')
yyaxis right
try
    plot(dff{days_to_plot(5)}(commoncells(cellno,days_to_plot(5)),:),'g') 
end
title(sprintf('day %i', days_to_plot(5)))

linkaxes([ax1 ax2 ax3 ax4 ax5],'xy')
han=axes(fig,'visible','off'); 
han.Title.Visible='on';
han.XLabel.Visible='on';
han.YLabel.Visible='on';
ylabel(han,'Y position');
xlabel(han,'Frames');
title(han,sprintf('Cell no. %03d', cellno));
savefig(sprintf('Z:\\cellregtest_behavior\\cell_%05d_days%02d_%02d_%02d_%02d_%02d.fig', cellno, days_to_plot))

%%
% align to behavior (rewards and solenoid) for each cell?
% per day, get this data...
range=5;
bin=0.2;
ccbinnedPerireward=cell(1,length(days));
ccrewdFF=cell(1,length(days));
for d=1:length(days)
    day=days(d);day=day{1};
    rewardsonly=day.rewards==1;
    cs=day.rewards==0.5;
    % runs for all cells
    [binnedPerireward,allbins,rewdFF] = perirewardbinnedactivity(dff{d}',rewardsonly,day.timedFF,range,bin); %rewardsonly if mapping to reward
    % now extract ids only of the common cells
    ccbinnedPerireward{d}=binnedPerireward(commoncells(:,d),:);
    ccrewdFF{d}=rewdFF(:,commoncells(:,d),:);
end
%%
% plot
%cellno=34; % cell to align
optodays=[];%5,6,7,9,10,11,13,14,16,17,18];
for cellno=1:length(commoncells) %or align all cells hehe
    figure;
    for d=1:length(days)
        pltrew=ccbinnedPerireward{d};
        if ~any(optodays(:)==d)            
            plot(pltrew(cellno,:)', 'Color', 'black')            
        else
            plot(pltrew(cellno,:)', 'Color', 'red')    
        end
        hold on;        
    end
    % plot reward location as line
    xticks([1:5:50, 50])
    x1=xline(median([1:5:50, 50]),'-.b','Reward'); %{'Conditioned', 'stimulus'}
    xticklabels([allbins(1:5:end) range]);
    xlabel('seconds')
    ylabel('dF/F')
    title(sprintf('Cell no. %04d', cellno))
end