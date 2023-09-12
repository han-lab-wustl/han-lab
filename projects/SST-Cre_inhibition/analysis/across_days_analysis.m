% across days analysis
% plot dff of certain cells across opto epoch (ep2)
cells_plot = [7 13 15 19 32 43 44 59 77 135];
% cells_plot = randi([1 length(cc)],1,20);
opto = [1 4 7 10 13 16 19]; % ep 2
opto = [3 6 9 12 15 18 20]; % ctrl

clear legg;
for cellno=cells_plot
    dd=1; %for legend
    figure;
    clear legg;
    for d=opto
        day=days(d);day=day{1};
        eps = find(day.changeRewLoc);
        eps = [eps length(day.changeRewLoc)]; % includes end of recording as end of a epoch
        % ep 2
        eprng = eps(2):eps(3);
    %     mask = (day.trialnum(eprng)>=3) & (day.trialnum(eprng)<8); % only first 5 trials
        mask = day.trialnum(eprng)>=8; % non opto trials
        rng = eprng(mask);
        try %if cell exists on that day, otherwise day is dropped...
            plot(day.dFF(rng,cc(cellno,d)))                        
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
    title(sprintf('Control day, Cell no. %04d', cellno))
end

%%
% get dff in opto period per cell across opto days vs. control days
opto = [1 4 7 10 13 16 19]; % ep 2
ctrl = [3 6 9 12 15 18 20]; % ctrl
dff_day_on = NaN(length(cc),length(opto));
dff_day_off = NaN(length(cc),length(opto));
dd=1; % for day counts
for d=opto    
    day=days(d);day=day{1};
    eps = find(day.changeRewLoc);
    eps = [eps length(day.changeRewLoc)]; % includes end of recording as end of a epoch
    % ep 2
    eprng = eps(2):eps(3);
    mask = (day.trialnum(eprng)>=3) & (day.trialnum(eprng)<8); % only first 5 trials
    rng_on = eprng(mask);
    mask = day.trialnum(eprng)>=8; % non opto trials
    rng_off = eprng(mask);
    [dFF,bordercells_b] = remove_border_cells_all_cells(day.stat, day.dFF);    
    cell_day=[1:size(day.dFF,2)];
    bordercells = cell_day(logical(bordercells_b));
    for cellno=1:length(cc)
        if cc(cellno,d)>0 && ~ismember(cc(cellno,d),bordercells)%if cell exists on that day, otherwise day is dropped...
            dff_day_on(cellno,dd) = mean(dFF(rng_on,cc(cellno,d)),'omitnan');                           
            dff_day_off(cellno,dd) = mean(dFF(rng_off,cc(cellno,d)),'omitnan'); 
        end
    end
    dd=dd+1;
end   
dd=1;
dff_ctrl_on = NaN(length(cc),length(ctrl));
dff_ctrl_off = NaN(length(cc),length(ctrl));
for d=ctrl    
    day=days(d);day=day{1};
    eps = find(day.changeRewLoc);
    eps = [eps length(day.changeRewLoc)]; % includes end of recording as end of a epoch
    % ep 2
    eprng = eps(2):eps(3);
    mask = (day.trialnum(eprng)>=3) & (day.trialnum(eprng)<8); % only first 5 trials
    rng_on = eprng(mask);
    mask = day.trialnum(eprng)>=8; % non opto trials
    rng_off = eprng(mask);
    [dFF,bordercells_b] = remove_border_cells_all_cells(day.stat, day.dFF);    
    cell_day=[1:size(day.dFF,2)];
    bordercells = cell_day(logical(bordercells_b));
    for cellno=1:length(cc)
        if cc(cellno,d)>0 && ~ismember(cc(cellno,d),bordercells) %if cell exists on that day, otherwise day is dropped...
            dff_ctrl_on(cellno,dd) = mean(dFF(rng_on,cc(cellno,d)),'omitnan');                           
            dff_ctrl_off(cellno,dd) = mean(dFF(rng_off,cc(cellno,d)),'omitnan'); 
        end
    end
    dd=dd+1;
end  
%%
figure;
% remove cells 
diff_opto = dff_day_on-dff_day_off;
% diff_opto = diff_opto(mean(diff_opto,2,'omitnan')>=0,:); % temp filter
diff_ctrl = dff_ctrl_on-dff_ctrl_off;
% diff_ctrl = diff_ctrl(mean(diff_opto,2,'omitnan')>=0,:); % temp filter
plot(1,mean(diff_opto,2,'omitnan'), 'ro'); hold on;
plot(2,mean(diff_ctrl,2,'omitnan'), 'bo')
for cellno=1:length(diff_opto)
    plot([1 2], [mean(diff_opto(cellno,:),2,'omitnan') mean(diff_ctrl(cellno,:),2,'omitnan')],'k')
end

xlim([0 3])
xticks([1 2])
[h,p,~,stat] = ttest(mean(diff_opto,2,'omitnan'),mean(diff_ctrl,2,'omitnan'))
