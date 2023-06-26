%% COM general view
%EH 201219
% to do:
% bonferroni correction or using ANOVA with post hoc?
% change labels to "first x trials" and "last y trials". insert variable
% fit learning curve to mean lick graph, print values
% save values and figures
% keeping better track of no lick trials, know which are dropped.

% generates three plots
% COM plot with some modifications
% ratio of peri-rewad licks to non-peri
% lick distance from rew start done by lick, not trial.
% for ratio and lick dist, removes no lick trials for t-test
% added bunch of variables to control number of trials and peri-rew window
%add quick lick correction with diff to eliminate consecutive "licks"
% Now lick distance from start of rew zone (=0)


% bugs:
% remove trials with no licks

% This script plot the mouse behavior and the center of mass (COM) analysis.
% COM is simply the mean of the position of lick along the track for each trial.
% Normalized COM is the COM minus the center of the reward location. COM =
% 0 is at the center of the reward location.
% The imput required is the VR scructure collected from the VR computer.
% It's designed for sessions having at least 3 reward locations.
% All the VR files after july 2020 are working for this script.
% Running the file, will ask you to pick up the file that you want to
% analyze. the workspace can be empty.

%%-----edits--------
%12/7 edit- fixed an error in which licks were being saved as one iteration
%earlier. fixed an error in which the first trial was excluded when non
%probes occured. added a feature of checking for matlab version to use
%sgtitle instead of the required additional function mtit


%12/21 edit- Adding Speed analysis On top of lick analysis. Lines 91-95 for
%new velocity variables. removed hard coding of number of probels on line
%127-8
%added ROE,meanROE,stdROE,bincount, and other Roe related variables tagged
%with GM in front. Added two additional sections of velocity analysis, one
%for mean roe accross trials, one for comparing accross epochs. coded in
%conversion rate

%02/16 edit- consecutive detected licks are removed from the raw data plot

%08/13/21 edit - converted ROE into actual velocity for analysis purposes
%(just after loading)

% NB: mtit function is required for matlab 2017b.
%% ------------------------------------------------------------------------
clc
clear %clean up the environment
close all;
mouse_name = "e201";
fltype = 'daypth'; %vs. day path
if fltype == 'select'
    [filename,filepath] = uigetfile('*.mat','MultiSelect','on');
    days = filename;
else
    % ZD added for loop for multiple days
    days = [55:73,75:80];
    src = 'Z:\sstcre_imaging';
end
ill = {}; % init
for dy=1:length(days)        
    if fltype == 'select'
        clearvars -except filepath filename dy days fltype
        file = fullfile(filepath,filename{dy});
    else        
        clearvars -except mouse_name days src dy fltype % Zahra added for for loop
        daypth = dir(fullfile(src, mouse_name, string(days(dy)), "behavior", "vr\*.mat"));
        filename{dy} = daypth.name;
        filepath{dy} = daypth.folder;
        file=fullfile(filepath{dy},filename{dy});
    end
    load(file) %load it
    if length(find(VR.changeRewLoc)) > 1
    try % based on 2 diff ways to read files
        cd (filepath{dy}); %EH set path
    catch 
        cd (filepath); 
    end
    addpath('C:\Users\Han\Documents\MATLAB\han-lab\hidden_reward_zone_task\behavior'); % ZD code path
    COMgeneralanalysis
    
    
    figure;
    %raw data plot
    subplot(3,8,[1 2 9 10 17 18])
    plottitle = [VR.name_date_vr ' lick distance analysis']; %name of the file you are analyzing
    if eval(version)>= 18 %checks if version of matlab is current enough to use inbed function
       sgtitle(plottitle,'fontsize',14,'interpreter','none')
    else %requires a function mtit
    mtit(plottitle,'fontsize',14,'interpreter','none') %name of the file you are analising as main title
    end
    scatter(time_min,ypos,1,'.','MarkerEdgeColor',[0.6 0.6 0.6])
    hold on
    scatter(islickX,islickYpos,10,'r','filled')
    scatter(rewX,rewYpos,10,'b','filled')
    for mm = 1:length(changeRewLoc)-1 %the rectangle indicating the reward location, overlaps the probe trials referring to the previous reward location
        rectangle('position',[time_min(changeRewLoc(mm)) RewLoc(mm)-rewSize time_min(changeRewLoc(mm+1))-time_min(changeRewLoc(mm)) 2*rewSize],'EdgeColor',[0 0 0 0],'FaceColor',[0 .5 .5 0.3])
    end
    ylabel('track position - cm')
    xlabel('time - minutes')
    title ('lick (red) along track - and reward (blue)')
    
    %COM plots
    for tt = 1:length(RewLoc) % for the three reward locations..
        if tt<=3 && ~isempty(allCOM{tt})
            subplot(3,8,[3+8*(tt-1) 4+8*(tt-1)]) % blue normalized lick scatterplot all the trials, succesfull or not are considered
            hold on
%             line([1,length(allCOM{tt})],[-7.5,-7.5],'LineStyle','--','LineWidth',1.5,'Color',	[0 0 0 0.4]) % dashed line is plotted at the nomalized COM (y=0 always) - 7.5
             line([1,length(allCOM{tt})],[0,0],'LineStyle','--','LineWidth',1.5,'Color',	[0 0 0 0.4]) %EH line at 0 for start of rew zone
            for i=1:numel(trialyposlicks{tt}) % the x axis indicates the trial number 
                % (eg. if trial 4 has no licks, the location 4 on the x axes will appear empty and the next trial (trial 5)
                % will be plotted at x positon 5)
                scatter(ones(size(trialyposlicks{tt}{i})).*i,(trialyposlicks{tt}{i})-RewLocStart(tt),38,[0 0.4470 0.7410],'filled')
                xlabel('trial no.')
                ylabel('distance to rew loc')
            end
            alpha(.05)
            scatter((1:length(allCOM{tt})),allCOM{tt},50,'g','filled')
            scatter(find(failure{tt}),allCOM{tt}(find(failure{tt})),50,'r','filled') 
            %failed trials are plotted with a red dot.
%             legend({'licks', 'successes', 'failures'})
            subplot(3,8,[5+8*(tt-1) 6+8*(tt-1)])% black normalized lick scatterplot with std. 
            % all the trials, succesfull or not are considered
            errorbar(allCOM{tt},allstdCOM{tt},'k.','LineWidth',1.5,'CapSize',0,'MarkerFaceColor','k','MarkerSize',25);
            hold on
%             line([1,length(allCOM{tt})],[-7.5,-7.5],'LineStyle','--','LineWidth',
% 1.5,'Color',	[0 0 0 0.4])
            line([1,length(allCOM{tt})],[0,0],'LineStyle','--','LineWidth',1.5,'Color',	[0 0 0 0.4])%EH line at 0 for start of rew zone
            title([ 'reward location ' num2str(tt) ' = ' num2str(RewLocStart(tt))])
            
            if sum(abs(allCOM{tt})>0)>10
                subplot(3,8, [7+8*(tt-1) 8+8*(tt-1)]) % barplot with p value of first and last five trials. only succesfull trials are considered
                succ = find(abs(COM{tt})>0);
                
%                 y=[(mean(allRatio{tt}(succ(1:numTrialsStart)))) (mean(allRatio{tt}(succ(end-(numTrialsEnd-1):end))))];%EH only successes
%                 [h,p1]=ttest2((allRatio{tt}(succ(1:numTrialsStart))),(allRatio{tt}(succ(end-(numTrialsEnd-1):end)))); %perform t-test. %EH
%               %ratio  
%                 y=[(nanmean(allRatio{tt}(1:numTrialsStart))) (nanmean(allRatio{tt}(end-(numTrialsEnd-1):end)))];
%                 [h,p1]=ttest2((allRatio{tt}(1:numTrialsStart)),(allRatio{tt}(end-(numTrialsEnd-1):end))); %perform t-test
                
                %distance of licks 
                lickDistTrim=[];
                keepLickDist=[];
                earlyLickDist=[];
                lateLickDist=[];
                lickDistTrim=lickDist(tt,:);
                keepLickDist = any(~cellfun('isempty',lickDistTrim), 1);  %// keep columns that don't only contain []
                lickDistTrim = lickDistTrim(:,keepLickDist);
                earlyLickDist=cat(2,lickDistTrim{1,1:numTrialsStart});
                lateLickDist=cat(2,lickDistTrim {1,end-(numTrialsEnd-1):end});
                
                y=[(mean(earlyLickDist)) (mean(lateLickDist))];
                [h,p1]=ttest2(earlyLickDist,lateLickDist); %perform t-test
                
                %EH plot ratio
%                 y=[(nanmean(allRatio{tt}(1:numTrialsStart))) (nanmean(allRatio{tt}(end-(numTrialsEnd-1):end)))];
%                 [h,p1]=ttest2((allRatio{tt}(1:numTrialsStart)),(allRatio{tt}(end-(numTrialsEnd-1):end))); %perform t-tes
                
%                 y=[abs(nanmean(COM{tt}(succ(1:5)))) abs(nanmean(COM{tt}(succ(end-4:end))))]; %only succesfull trials considered
%                 [h,p1]=ttest(COM{tt}(succ(1:5)),COM{tt}(succ(end-4:end))); %perform t-test
                hBar=bar(y);
                Labels = {'first five trials', 'last five trials'};
                set(gca,'XTick', 1:2, 'XTickLabel', Labels);
                ctr2 = bsxfun(@plus, hBar(1).XData, [hBar(1).XOffset]');
                if p1<0.05
                    hold on
                    plot(ctr2(1:2), [1 1]*y(1,1)*1.1, '-k', 'LineWidth',2)
                    plot(mean(ctr2(1:2)), y(1,1)*1.15, '*k')
                    hold off
                end
                text(mean(ctr2(1:2))+0.3,y(1,1)*1.15,['p = ' num2str(round(p1,6))])
            else
                disp('Not enough succesfull trials in the last reward zone to compute t-test');
            end
        end
    end
    % save inter lick interval - ZD
    ill{dy} = InterLickInterval;
    
    % %% InterLick Interval and Total Lick Count
    % epochs x max trials x max inter lick intervals
    figure; 
    x = 1:2;
    numtrialcompair = 5;
    for i = 1:size(InterLickInterval,1)
        successful = find(failure{i} == 0);
    subplot(2,size(InterLickInterval,1),i)
    imagesc(flipud(squeeze(InterLickInterval(i,:,:))'))
    xlabel('trial number')
    y1 = yticklabels;
    yticklabels(flipud(y1))
    ylabel('Interlick Interval Number')
    if length(successful) >= numtrialcompair*2
    earlyInterval = reshape(InterLickInterval(i,successful(1:numtrialcompair),:),1,numtrialcompair*size(InterLickInterval,3));
    lateInterval = reshape(InterLickInterval(i,successful(end-numtrialcompair+1:end),:),1,numtrialcompair*size(InterLickInterval,3));
    earlyInterval(isnan(earlyInterval)) = [];
    lateInterval(isnan(lateInterval)) = [];
    earlymean = nanmean(earlyInterval);
    latemean = nanmean(lateInterval);
    earlystd =nanstd(earlyInterval)/sqrt(length(earlyInterval));
    latestd = nanstd(lateInterval)/sqrt(length(lateInterval));
    subplot(2,size(InterLickInterval,1),i+size(InterLickInterval,1))
    bar(x,[earlymean latemean])
    hold on
    er = errorbar(x,[earlymean...
        latemean]...
        ,[earlystd latestd],[earlystd latestd]);
    er.Color = [0 0 0];
    [h,p] = ttest2(earlyInterval,lateInterval);
    text(1.5,max([earlymean latemean] +0.001),['t-test2 p = ' num2str(p)])
           Labels = {['first ' num2str(numtrialcompair) ' trials'], ['last ' num2str(numtrialcompair) ' trials']};
                    set(gca,'XTick', 1:2, 'XTickLabel', Labels);
    hold off
    end
    
    end
end
dst = 'Y:\sstcre_analysis\hrz\lick_analysis';
save(fullfile(dst, sprintf('%s_day%i-%i_interlickintervals.mat', mouse_name, ...
    min(days), max(days))), "ill")

%%
% 
function licks = correct_artifact_licks(ybinned,licks)
    % delete consecutive licks from signal
    x = 3; % here you can modify the cm

    % Take the difference (slope between points)
    diffL = diff(licks) == 1 ;
    
    % Pad zero out front
    diffL = [0 diffL];
    
    % keep only the starting point of the lick transients
    licks = licks.* logical(diffL);
    
    % delete all the licks before 'x' cm
    licks(ybinned<=x) = 0; 
end

