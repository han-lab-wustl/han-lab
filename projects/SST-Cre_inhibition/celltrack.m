% Zahra
% get cells detected in cellreg and do analysis

clear all
src = 'Y:\sstcre_analysis\'; % main folder for analysis
animal = 'e201';
weekfld = 'days1-18';
pth = dir(fullfile(src, "celltrack", sprintf([animal, '_', weekfld]), "Results\*cellRegistered*"));
load(fullfile(pth.folder, pth.name))
% find cells in all sessions
[r,c] = find(cell_registered_struct.cell_to_index_map~=0);
[counts, bins] = hist(r,1:size(r,1));
sessions=length(cell_registered_struct.centroid_locations_corrected);% specify no of sessions
cindex = bins(counts==12); % finding cells AT LEAST 2 SESSIONS???
commoncells=zeros(length(cindex),sessions);
for ci=1:length(cindex)
    commoncells(ci,:)=cell_registered_struct.cell_to_index_map(cindex(ci),:);
end
 
% load mats from all days
fls = dir(fullfile(src, 'fmats', sprintf('%s\\%s*.mat', animal, animal)));%dir('Z:\cellreg1month_Fmats\*YC_Fall.mat');
days = cell(1, length(fls));
for fl=1:length(fls)
    day = fls(fl);
    days{fl} = load(fullfile(day.folder,day.name));
end
cc=commoncells;
% calculate dff across all
dff = cell(1,sessions);
for i=1:sessions
    dff{i+2} = redo_dFF(days{i}.F, 31.25, 20, days{i}.Fneu);
    disp(i)
end
% save(fullfile(src, "celltrack", sprintf([animal, '_', weekfld]), "Results\dff.mat"), 'dff', '-v7.3')
load(fullfile(src, "celltrack", sprintf([animal, '_', weekfld]), "Results\dff.mat"))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
ctab = hsv(length(cc));

for i=1:20
    %multi plot of cell mask across all 5 days
    figure(i); 

    axes=cell(1,sessions);
    for ss=1:sessions        
        day=days(ss);day=day{1};
        axes{ss}=subplot(3,4,ss); % 2 rows, 3 column, 1 pos; 20 days
        imagesc(day.ops.meanImg) %meanImg or max_proj
        colormap('gray')
        hold on;
        try
            plot(day.stat{1,cc(i,ss)}.xpix, day.stat{1,cc(i,ss)}.ypix, 'Color', [ctab(i,:) 0.3]);
        end
        axis off
        title(sprintf('day %i', ss)) %sprintf('day %i', ss)
        %title(axes{ss},sprintf('Cell %0d4', i))
    end
    linkaxes([axes{:}], 'xy')
    %savefig(sprintf("Z:\\suite2pconcat1month_commoncellmasks\\cell_%03d.fig",i+250)) %changed to reflect subset of cells plotted
end

%%
% align all cells across all days in 1 fig
% colormap to iterate thru
ctab = hsv(length(cc));
figure;
axes=zeros(1,sessions);
for ss=1:sessions
    day=days(ss);day=day{1};
    axes(ss)=subplot(4,4,ss);%(4,5,ss); % 2 rows, 3 column, 1 pos; 20 days
    imagesc(day.ops.meanImg)
    colormap('gray')
    hold on;
    for i=1:length(commoncells)
        try
            plot(day.stat{1,cc(i,ss)}.xpix, day.stat{1,cc(i,ss)}.ypix, 'Color', [ctab(i,:) 0.3]);
        end
    end
    axis off
    [dy,~] = fileparts(day.ops.data_path);
    [~,dynm] = fileparts(dy);
    dy=str2num(dynm);
    title(sprintf('day %i', dy))
end
linkaxes(axes, 'xy')
%savefig(sprintf("Z:\\202300201cells.fig"))

%%
% plot traces across all days
cellno=78;
cells_to_loop = 20; %length(cc)
for cellno=1:cells_to_loop %no. of common cells
    grayColor = [.7 .7 .7];    
    fig=figure;
    for dayplt=3:sessions
        ax1=subplot(sessions,1,dayplt);
        day=days(dayplt);day=day{1};
        try
            plot(dff{dayplt}(cc(cellno,dayplt),:),'k') % 2 in the first position is cell no
        end
        axs{dayplt}=ax1;
        set(gca,'XTick',[], 'YTick', [])
        set(gca,'visible','off')
    end
    % linkaxes([axs{:}],'xy')
    copygraphics(gcf, 'BackgroundColor', 'none');
    title(sprintf('Cell no. %03d', cellno));
end
%%
%only load if saved data from previous run
% load('Z:\\dff_221206-30.mat')
% % load mats from all days
% fls = dir(fullfile('Z:\cellreg1month_Fmats\', '**\*YC_Fall.mat'));%dir('Z:\cellreg1month_Fmats\*YC_Fall.mat');
% days = cell(1, length(fls));
% for fl=1:length(fls)
%     day = fls(fl);
%     days{fl} = load(fullfile(day.folder,day.name));
% end

% align to behavior (rewards and solenoid) for each cell?
% per day, get this data...
range=5;
bin=0.2;
ccbinnedPerireward=cell(1,length(days));
ccrewdFF=cell(1,length(days));
for d=1:length(days)
    day=days(d);day=day{1};
    try
        rewardsonly=day.rewards>1;
        cs=day.rewards==0.5;
        % runs for all cells
        [binnedPerireward,allbins,rewdFF] = perirewardbinnedactivity(dff{d}',rewardsonly,day.timedFF,range,bin); %rewardsonly if mapping to reward
        % now extract ids only of the common cells
        ccbinnedPerireward{d}=binnedPerireward; %changes to reflect when mapping is not present for all days
        ccrewdFF{d}=rewdFF;
    end
end
%%
% plot
cellno=2; % cell to align
% optodays=[5,6,7,9,10,11,13,14,16,17,18];
for cellno=1:length(cc) %or align all cells hehe
    figure;
    daynms = zeros(1, length(days));
    for d=1:length(days)
        pltrew=ccbinnedPerireward{d};
%         if ~any(optodays(:)==d)            
           try
               plot(pltrew(cc(cellno,d),:)') %important distinction
               [daynm,~] = fileparts(days{d}.ops.data_path);
               [~,daynm] = fileparts(daynm);
               daynms(d)=str2num(daynm);
           end
%         else
%             plot(pltrew(cellno,:)', 'Color', 'red')    
%         end
        hold on;        
    end
    % plot reward location as line
    xticks([1:5:50, 50])
    x1=xline(median([1:5:50, 50]),'-.b','Reward'); %{'Conditioned', 'stimulus'}
    xticklabels([allbins(1:5:end) range]);
    xlabel('seconds')
    ylabel('dF/F')
    daynms=daynms(daynms~=0) ;
    legend(num2str([daynms(:)])) %plots days that cell was detected
    title(sprintf('Cell no. %04d', cellno))
end