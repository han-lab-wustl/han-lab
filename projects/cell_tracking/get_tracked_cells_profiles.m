% Zahra
% get cells detected in cellreg and do analysis
clear all;
% find cells detected in all 4 weeks (transform 1)
% we want to keep all these cells
animal = 'e201';%e200';
load([path to folder]"commoncells_atleastoneactivedayperweek_4weeks_week2daymap.mat")
%%

% load mats from all days
fls = dir([path to mouse folder] '*day*_Fall.mat'));
days = cell(1, length(fls));
for fl=1:length(fls)
    disp(fl);
    dy = fls(fl);
    days{fl} = load(fullfile(dy.folder,dy.name));
end
cc=cellmap2dayacrossweeks;
sessions_total=length(days);

%%
% plot F (and ideally dff) over ypos
days_to_plot=[1,2,3,4,5];%1:sessions_total; %[1 4 7 10 13 16]; %plot 5 days at a time
% picks a random cell to plot
cellno=randi([1 length(cc)],1,1);
grayColor = [.7 .7 .7];
fig=figure;
subplot_j=1;
for dayplt=days_to_plot
    ax1=subplot(length(days_to_plot),1,subplot_j);
    day=days(dayplt);day=day{1};
    plot(day.ybinned, 'Color', grayColor); hold on; 
    plot(day.changeRewLoc, 'b')
    plot(find(day.licks),day.ybinned(find(day.licks)),'r.')     
    rew = day.rewards>0.5; % codes for single or double rewards
    plot(find(rew),day.ybinned(find(rew)),'b.', ...
        'MarkerSize',10); 
    yyaxis right
    try
        plot(day.dFF(:,cc(cellno,dayplt)),'g') % 2 in the first position is cell no
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
sgtitle(sprintf('Cell no. %03d', cellno));
savefig(sprintf('Z:\\cellregtest_behavior\\cell_%05d_days%02d_%02d_%02d_%02d_%02d.fig', cellno, days_to_plot))
%% peri velocity analysis

range=5; % number of seconds before and after start
bin=0.1; % s

%only get days with rewards (exclude training)
daysrewards = [1 2 3 4 5];
ccbinnedPerivelocity=cell(1,length(daysrewards));
ccveldFF=cell(1,length(daysrewards));
for d=daysrewards
    day=days(d);day=day{1};    
    rewardsonly=day.rewards==1;
    % filter by epoch optoed
    eps = find(day.changeRewLoc);
    eps = [eps length(day.changeRewLoc)]; % includes end of recording as end of a epoch
    
    % only get non rewarded stops in current config
    [binnedPerivelocity,allbins,veldFF] = perivelocitybinnedactivity(day.forwardvel, ...
        rewardsonly, day.dFF, day.timedFF,range,bin,1); %rewardsonly if mapping to reward
    % now extract ids only of the common cells
    ccbinnedPerivelocity{d}=binnedPerivelocity;
    ccveldFF{d}=veldFF;
end

%%
% PLOT START TRIGGERED CELL ACTIVITY
% plot dff of certain cells across opto epoch (ep2)
% cells_to_plot = randi([1 1317],1,100);
cells_to_plot = randi([1 size(cc,1)],1,20);%[7 13 15 19 32 43 44 59 77 135];
%[161 157 152 140 135 106 93 77 59 58 53 52 48 47 44 43 42 41 33 32 27 19 15 13 12 8 7 6 4];%[1:200];
for cellno=cells_to_plot
    dd=1; %for legend
    figure;
    clear legg;
    for d=daysrewards
        clear pltrew;
        pltvel=ccbinnedPerivelocity{d};
        try %if cell exists on that day, otherwise day is dropped...
%             if ismember(d,opto)
                plot(pltvel(cc(cellno,d),:)','k'); hold on;           
%             else
%                 plot(pltrew(cc(cellno,d),:)','k')            
%             end
%             else
%               plot(pltrew(cc(cellno,d),:)', 'Color', 'red')    
%             end
                legg{dd}=sprintf('day %d',d); dd=dd+1;           
        end
        hold on;        
    end   
     % plot reward location as line
%     xticks([1:5:50, 50])
%     x1=xline(median([1:5:50, 50]),'-.b','Reward'); %{'Conditioned', 'stimulus'}
%     xticklabels([allbins(1:5:end) range]);
    xlabel('seconds')
    ylabel('dF/F')
    legend(char(legg))
    title(sprintf('Cell no. %04d', cellno))
end
