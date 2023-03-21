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
[filename,filepath] = uigetfile('*.mat','MultiSelect','off');
file = [filepath filename];
load(file) %load it
if length(find(VR.changeRewLoc)) > 1
cd (filepath); %EH set path

COMgeneralanalysis
    
%  %% Mean lick by trials plot
%     
%     figure;
%     %raw data plot
%     subplot(3,8,[1 2 9 10 17 18])
%     if eval(version)>= 18 %checks if version of matlab is current enough to use inbed function
%        sgtitle([VR.name_date_vr ' Mean lick position by trial'],'fontsize',14,'interpreter','none')
%     else %requires a function mtit
%     mtit([VR.name_date_vr ' Mean lick position by trial'],'fontsize',14,'interpreter','none') %name of the file you are analising as main title
%     end
%     scatter(time_min,ypos,1,'.','MarkerEdgeColor',[0.6 0.6 0.6])
%     hold on
%     scatter(islickX,islickYpos,10,'r','filled')
%     scatter(rewX,rewYpos,10,'b','filled')
%     for mm = 1:length(changeRewLoc)-1 %the rectangle indicating the reward location, overlaps the probe trials referring to the previous reward location
%         rectangle('position',[time_min(changeRewLoc(mm)) RewLoc(mm)-rewSize time_min(changeRewLoc(mm+1))-time_min(changeRewLoc(mm)) 2*rewSize],'EdgeColor',[0 0 0 0],'FaceColor',[0 .5 .5 0.3])
%     end
%     ylabel('track position - cm')
%     xlabel('time - minutes')
%     title ('lick (red) along track - and reward (blue)')
%     
%     %COM plots
%     for tt = 1:length(RewLoc) % for the three reward locations..
%         if tt<=3 && ~isempty(allCOM{tt})
%             subplot(3,8,[3+8*(tt-1) 4+8*(tt-1)]) % blue normalized lick scatterplot all the trials, succesfull or not are considered
%             hold on
%             line([1,length(allCOM{tt})],[-rewSize,-rewSize],'LineStyle','--','LineWidth',1.5,'Color',	[0 0 0 0.4]) % dashed line is plotted at the nomalized COM (y=0 always) - 7.5
%             for i=1:numel(trialyposlicks{tt}) % the x axis indicates the trial number (eg. if trial 4 has no licks, the location 4 on the x axes will appear empty and the next trial (trial 5) will be plotted at x positon 5)
%                 scatter(ones(size(trialyposlicks{tt}{i})).*i,(trialyposlicks{tt}{i})-RewLoc(tt),38,[0 0.4470 0.7410],'filled')
%             end
%             alpha(.05)
%             scatter((1:length(allCOM{tt})),allCOM{tt},50,'filled')
%             scatter(find(failure{tt}),allCOM{tt}(find(failure{tt})),50,'r','filled') %failed trials are plotted with a red dot.
%             
%             subplot(3,8,[5+8*(tt-1) 6+8*(tt-1)])% black normalized lick scatterplot with std. all the trials, succesfull or not are considered
%             errorbar(allCOM{tt},allstdCOM{tt},'k.','LineWidth',1.5,'CapSize',0,'MarkerFaceColor','k','MarkerSize',25);
%             hold on
%             line([1,length(allCOM{tt})],[-rewSize,-rewSize],'LineStyle','--','LineWidth',1.5,'Color',	[0 0 0 0.4])
%             title([ 'reward location ' num2str(tt) ' = ' num2str(RewLoc(tt))])
%             
%             if sum(abs(allCOM{tt})>0)>10 && ~isempty(COM{tt})
%                 subplot(3,8, [7+8*(tt-1) 8+8*(tt-1)]) % barplot with p value of first and last five trials. only succesfull trials are considered
%                 succ = find(abs(COM{tt})>0);
%                 
%                                  
%                 y=[abs(nanmean(COM{tt}(succ(1:numTrialsStart)))) abs(nanmean(COM{tt}(succ(end-(numTrialsEnd-1):end))))]; %only succesfull trials considered
% %                 [h,p1]=ttest(COM{tt}(succ(1:numTrialsStart)),COM{tt}(succ(end-(numTrialsEnd-1):end))); %perform t-test
%                 [h,p1]=ttest2(COM{tt}(succ(1:numTrialsStart)),COM{tt}(succ(end-(numTrialsEnd-1):end))); %EH perform unpaired t-test
%                 hBar=bar(y);
%                 Labels = {'first five trials', 'last five trials'};
%                 set(gca,'XTick', 1:2, 'XTickLabel', Labels);
%                 ctr2 = bsxfun(@plus, hBar(1).XData, [hBar(1).XOffset]');
%                 if p1<0.05
%                     hold on
%                     plot(ctr2(1:2), [1 1]*y(1,1)*1.1, '-k', 'LineWidth',2)
%                     plot(mean(ctr2(1:2)), y(1,1)*1.15, '*k')
%                     hold off
%                 end
%                 text(mean(ctr2(1:2))+0.3,y(1,1)*1.15,['p = ' num2str(round(p1,3))])
%             else
%                 disp('Not enough succesfull trials in the last reward zone to compute t-test');
%             end
%         end
%     end   
%     
%  %% ratio plot
%     
%     figure;
%     %raw data plot
%     subplot(3,8,[1 2 9 10 17 18])
%     if eval(version)>= 18 %checks if version of matlab is current enough to use inbed function
%        sgtitle([VR.name_date_vr ' Peri-reward ratio'],'fontsize',14,'interpreter','none')
%     else %requires a function mtit
%     mtit([VR.name_date_vr ' Peri-reward ratio '],'fontsize',14,'interpreter','none') %name of the file you are analising as main title
%     end
%     scatter(time_min,ypos,1,'.','MarkerEdgeColor',[0.6 0.6 0.6])
%     hold on
%     scatter(islickX,islickYpos,10,'r','filled')
%     scatter(rewX,rewYpos,10,'b','filled')
%     for mm = 1:length(changeRewLoc)-1 %the rectangle indicating the reward location, overlaps the probe trials referring to the previous reward location
%         rectangle('position',[time_min(changeRewLoc(mm)) RewLoc(mm)-rewSize time_min(changeRewLoc(mm+1))-time_min(changeRewLoc(mm)) 2*rewSize],'EdgeColor',[0 0 0 0],'FaceColor',[0 .5 .5 0.3])
%     end
%     ylabel('track position - cm')
%     xlabel('time - minutes')
%     title ('lick (red) along track - and reward (blue)')
%     
%     %COM plots
%     for tt = 1:length(RewLoc) % for the three reward locations..
%         if tt<=3 && ~isempty(allCOM{tt})
%             subplot(3,8,[3+8*(tt-1) 4+8*(tt-1)]) % blue normalized lick scatterplot all the trials, succesfull or not are considered
%             hold on
% %             line([1,length(allCOM{tt})],[-7.5,-7.5],'LineStyle','--','LineWidth',1.5,'Color',	[0 0 0 0.4]) % dashed line is plotted at the nomalized COM (y=0 always) - 7.5
%              line([1,length(allCOM{tt})],[0,0],'LineStyle','--','LineWidth',1.5,'Color',	[0 0 0 0.4]) %EH line at 0 for start of rew zone
%             for i=1:numel(trialyposlicks{tt}) % the x axis indicates the trial number (eg. if trial 4 has no licks, the location 4 on the x axes will appear empty and the next trial (trial 5) will be plotted at x positon 5)
%                 scatter(ones(size(trialyposlicks{tt}{i})).*i,(trialyposlicks{tt}{i})-RewLocStart(tt),38,[0 0.4470 0.7410],'filled')
%             end
%             alpha(.05)
%             scatter((1:length(allCOM{tt})),allCOM{tt},50,'filled')
%             scatter(find(failure{tt}),allCOM{tt}(find(failure{tt})),50,'r','filled') %failed trials are plotted with a red dot.
%             
%             subplot(3,8,[5+8*(tt-1) 6+8*(tt-1)])% black normalized lick scatterplot with std. all the trials, succesfull or not are considered
%             errorbar(allCOM{tt},allstdCOM{tt},'k.','LineWidth',1.5,'CapSize',0,'MarkerFaceColor','k','MarkerSize',25);
%             hold on
% %             line([1,length(allCOM{tt})],[-7.5,-7.5],'LineStyle','--','LineWidth',1.5,'Color',	[0 0 0 0.4])
%             line([1,length(allCOM{tt})],[0,0],'LineStyle','--','LineWidth',1.5,'Color',	[0 0 0 0.4])%EH line at 0 for start of rew zone
%             title([ 'reward location ' num2str(tt) ' = ' num2str(RewLocStart(tt))])
%             
%             if sum(abs(allCOM{tt})>0)>10
%                 subplot(3,8, [7+8*(tt-1) 8+8*(tt-1)]) % barplot with p value of first and last five trials. only succesfull trials are considered
%                 succ = find(abs(COM{tt})>0);
%                 
% %                 y=[(mean(allRatio{tt}(succ(1:numTrialsStart)))) (mean(allRatio{tt}(succ(end-(numTrialsEnd-1):end))))];%EH only successes
% %                 [h,p1]=ttest2((allRatio{tt}(succ(1:numTrialsStart))),(allRatio{tt}(succ(end-(numTrialsEnd-1):end)))); %perform t-test. %EH
% %               %ratio  
% %                 y=[(nanmean(allRatio{tt}(1:numTrialsStart))) (nanmean(allRatio{tt}(end-(numTrialsEnd-1):end)))];
% %                 [h,p1]=ttest2((allRatio{tt}(1:numTrialsStart)),(allRatio{tt}(end-(numTrialsEnd-1):end))); %perform t-test
%                 
%                 %ratio of peri-reward licks 
%                 ratioTrim=[];
%                 earlyRatio=[];
%                 lateRatio=[];
%                 
%                 ratioTrim=allRatio{tt,:};%pull tt from array
%                 ratioTrim=ratioTrim(~isnan(ratioTrim));%remove NaNs
%                 earlyRatio=ratioTrim(1:numTrialsStart);
%                 lateRatio=ratioTrim (end-(numTrialsEnd-1):end);
%                 
%                 %EH plot ratio
%                 y=[(mean(earlyRatio)) (mean(lateRatio))];
%                 [h,p1]=ttest2((earlyRatio),(lateRatio)); %perform t-test
%                 
% %                 y=[abs(nanmean(COM{tt}(succ(1:5)))) abs(nanmean(COM{tt}(succ(end-4:end))))]; %only succesfull trials considered
% %                 [h,p1]=ttest(COM{tt}(succ(1:5)),COM{tt}(succ(end-4:end))); %perform t-test
%                 hBar=bar(y);
%                 Labels = {'first five trials', 'last five trials'};
%                 set(gca,'XTick', 1:2, 'XTickLabel', Labels);
%                 ctr2 = bsxfun(@plus, hBar(1).XData, [hBar(1).XOffset]');
%                 if p1<0.05
%                     hold on
%                     plot(ctr2(1:2), [1 1]*y(1,1)*1.1, '-k', 'LineWidth',2)
%                     plot(mean(ctr2(1:2)), y(1,1)*1.15, '*k')
%                     hold off
%                 end
%                 text(mean(ctr2(1:2))+0.3,y(1,1)*1.15,['p = ' num2str(round(p1,3))])
%             else
%                 disp('Not enough succesfull trials in the last reward zone to compute t-test');
%             end
%         end
%     end    
     %% lick dist plot
    
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
            for i=1:numel(trialyposlicks{tt}) % the x axis indicates the trial number (eg. if trial 4 has no licks, the location 4 on the x axes will appear empty and the next trial (trial 5) will be plotted at x positon 5)
                scatter(ones(size(trialyposlicks{tt}{i})).*i,(trialyposlicks{tt}{i})-RewLocStart(tt),38,[0 0.4470 0.7410],'filled')
            end
            alpha(.05)
            scatter((1:length(allCOM{tt})),allCOM{tt},50,'filled')
            scatter(find(failure{tt}),allCOM{tt}(find(failure{tt})),50,'r','filled') %failed trials are plotted with a red dot.
            
            subplot(3,8,[5+8*(tt-1) 6+8*(tt-1)])% black normalized lick scatterplot with std. all the trials, succesfull or not are considered
            errorbar(allCOM{tt},allstdCOM{tt},'k.','LineWidth',1.5,'CapSize',0,'MarkerFaceColor','k','MarkerSize',25);
            hold on
%             line([1,length(allCOM{tt})],[-7.5,-7.5],'LineStyle','--','LineWidth',1.5,'Color',	[0 0 0 0.4])
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
    
%     %% Speed Mean Plots
%      
%     figure;
%     %raw data plot
%     subplot(3,8,[1 2 9 10 17 18])
%     if eval(version)>= 18 %checks if version of matlab is current enough to use inbed function
%        sgtitle([VR.name_date_vr ' Mean ROE by trial'],'fontsize',14,'interpreter','none')
%     else %requires a function mtit
%     mtit([VR.name_date_vr ' Mean ROE by trial'],'fontsize',14,'interpreter','none') %name of the file you are analising as main title
%     end
%     scatter(time_min,ypos,1,'.','MarkerEdgeColor',[0.6 0.6 0.6])
%     hold on
% 
%     
%     for xx = 1:numel(ROE) %plots the binned ROE overlaying the raw data
%         for yy = 1:numel(ROE{xx})
%             scatter(VR.time(binypos{xx}{yy}(~isnan(binypos{xx}{yy})))/60,VR.ypos(binypos{xx}{yy}(~isnan(binypos{xx}{yy}))),2*rewSize,ROE{xx}{yy}(~isnan(binypos{xx}{yy})),'filled')
%             colormap(flipud(parula))
%     %         colormap((viridis())) % for a personal choice of colormap,
%     %         but commented out because requires download of code
%             
%         end
%     end
%     scatter(islickX,islickYpos,10,'r','x')
%     scatter(rewX,rewYpos,10,'b','filled')
%     
%     for mm = 1:length(changeRewLoc)-1 %the rectangle indicating the reward location, overlaps the probe trials referring to the previous reward location
%         rectangle('position',[time_min(changeRewLoc(mm)) RewLoc(mm)-rewSize time_min(changeRewLoc(mm+1))-time_min(changeRewLoc(mm)) 2*rewSize],'EdgeColor',[0 0 0 0],'FaceColor',[0 .5 .5 0.3])
%     end
%     ylabel('track position - cm')
%     xlabel('time - minutes')
%     title ('lick (red) along track - and reward (blue)')
%     
%     %meanROE plots
%     for tt = 1:length(RewLoc) % for the three reward locations..
%         if tt<=3 && ~isempty(allCOM{tt})
%             subplot(3,8,[3+8*(tt-1) 4+8*(tt-1)]) %plot the first column
%             hold on
%             for i=1:numel(ROE{tt}) %plot all the ROE values
%                 scatter(ones(size(ROE{tt}{i})).*i,(ROE{tt}{i}),38,[0 0.4470 0.7410],'filled')
%             end
%             alpha(.05)
%             scatter((1:length(meanROE{tt})),meanROE{tt},50,'filled') %plot the mean on top
%             scatter(find(failure{tt}),meanROE{tt}(find(failure{tt})),50,'r','filled')
%             first5success = find(failure{tt} == 0,5,'first');
%             last5success = find(failure{tt} == 0,5,'last');
%             
%             
%             subplot(3,8,[5+8*(tt-1) 6+8*(tt-1)]) %plot the mean velocity with std Error
%             errorbar(meanROE{tt},stdROE{tt},'k.','LineWidth',1.5,'CapSize',0,'MarkerFaceColor','k','MarkerSize',25);
%             hold on
%             title([ 'reward location ' num2str(tt) ' = ' num2str(RewLoc(tt))])
%             
%             if sum(abs(allCOM{tt})>0)>10
%                 subplot(3,8, [7+8*(tt-1) 8+8*(tt-1)]) %plot a paired t-test of the average velocity
%                 y=[abs(nanmean(meanROE{tt}(first5success))) abs(nanmean(meanROE{tt}(last5success)))];
%                 [h,p1]=ttest2(meanROE{tt}(first5success),meanROE{tt}(last5success));
%                 hBar=bar(y);
%                 Labels = {'first five trials', 'last five trials'};
%                 set(gca,'XTick', 1:2, 'XTickLabel', Labels);
%                 ctr2 = bsxfun(@plus, hBar(1).XData, [hBar(1).XOffset]');
%                 if p1<0.05
%                     hold on
%                     plot(ctr2(1:2), [1 1]*y(1,1)*1.1, '-k', 'LineWidth',2)
%                     plot(mean(ctr2(1:2)), y(1,1)*1.15, '*k')
%                     hold off
%                 end
%                 text(mean(ctr2(1:2))+0.3,y(1,1)*1.15,['p = ' num2str(round(p1,3))])
% 
%             else
%                 disp('Not enough succesfull trials in the last reward zone to compute t-test');
%             end
%         end
%     end  
%     
%     
%        %% Speed Mean cut to only first portion in common Plots
%      
%     figure;
%     %raw data plot
%     subplot(3,8,[1 2 9 10 17 18])
%     if eval(version)>= 18 %checks if version of matlab is current enough to use inbed function
%        sgtitle([VR.name_date_vr ' Mean ROE by trial'],'fontsize',14,'interpreter','none')
%     else %requires a function mtit
%     mtit([VR.name_date_vr ' Mean ROE by trial'],'fontsize',14,'interpreter','none') %name of the file you are analising as main title
%     end
%     scatter(time_min,ypos,1,'.','MarkerEdgeColor',[0.6 0.6 0.6])
%     hold on
% 
%     
%     for xx = 1:numel(ROE) %plots the binned ROE overlaying the raw data
%         for yy = 1:numel(ROE{xx})
%             scatter(VR.time(binypos{xx}{yy}(~isnan(binypos{xx}{yy})))/60,VR.ypos(binypos{xx}{yy}(~isnan(binypos{xx}{yy}))),2*rewSize,ROE{xx}{yy}(~isnan(binypos{xx}{yy})),'filled')
%             colormap(flipud(parula))
%     %         colormap((viridis())) % for a personal choice of colormap,
%     %         but commented out because requires download of code
%             
%         end
%     end
%     scatter(islickX,islickYpos,10,'r','x')
%     scatter(rewX,rewYpos,10,'b','filled')
%     
%     for mm = 1:length(changeRewLoc)-1 %the rectangle indicating the reward location, overlaps the probe trials referring to the previous reward location
%         rectangle('position',[time_min(changeRewLoc(mm)) RewLoc(mm)-rewSize time_min(changeRewLoc(mm+1))-time_min(changeRewLoc(mm)) 2*rewSize],'EdgeColor',[0 0 0 0],'FaceColor',[0 .5 .5 0.3])
%     end
%     ylabel('track position - cm')
%     xlabel('time - minutes')
%     title ('lick (red) along track - and reward (blue)')
%     
%     %meanROE plots
%     for tt = 1:length(RewLoc) % for the three reward locations..
%         if tt<=3 && ~isempty(allCOM{tt})
%             subplot(3,8,[3+8*(tt-1) 4+8*(tt-1)]) %plot the first column
%             hold on
%             for i=1:numel(cutROE{tt}) %plot all the ROE values
%                 scatter(ones(size(cutROE{tt}{i})).*i,(cutROE{tt}{i}),38,[0 0.4470 0.7410],'filled')
%             end
%             alpha(.05)
%             scatter((1:length(meancutROE{tt})),meancutROE{tt},50,'filled') %plot the mean on top
%             scatter(find(failure{tt}),meancutROE{tt}(find(failure{tt})),50,'r','filled')
%             first5success = find(failure{tt} == 0,5,'first');
%             last5success = find(failure{tt} == 0,5,'last');
%             
%             
%             subplot(3,8,[5+8*(tt-1) 6+8*(tt-1)]) %plot the mean velocity with std Error
%             errorbar(meancutROE{tt},semcutROE{tt},'k.','LineWidth',1.5,'CapSize',0,'MarkerFaceColor','k','MarkerSize',25);
%             hold on
%             title([ 'reward location ' num2str(tt) ' = ' num2str(RewLoc(tt))])
%             
%             if sum(abs(allCOM{tt})>0)>10
%                 subplot(3,8, [7+8*(tt-1) 8+8*(tt-1)]) %plot a paired t-test of the average velocity
%                 y=[abs(nanmean(meancutROE{tt}(first5success))) abs(nanmean(meancutROE{tt}(last5success)))];
%                 [h,p1]=ttest2(meancutROE{tt}(first5success),meancutROE{tt}(last5success));
%                 hBar=bar(y);
%                 Labels = {'first five trials', 'last five trials'};
%                 set(gca,'XTick', 1:2, 'XTickLabel', Labels);
%                 ctr2 = bsxfun(@plus, hBar(1).XData, [hBar(1).XOffset]');
%                 if p1<0.05
%                     hold on
%                     plot(ctr2(1:2), [1 1]*y(1,1)*1.1, '-k', 'LineWidth',2)
%                     plot(mean(ctr2(1:2)), y(1,1)*1.15, '*k')
%                     hold off
%                 end
%                 text(mean(ctr2(1:2))+0.3,y(1,1)*1.15,['p = ' num2str(round(p1,3))])
% 
%             else
%                 disp('Not enough succesfull trials in the last reward zone to compute t-test');
%             end
%         end
%     end  
% %     %% Analysis of speed distribution between Epoch 1 and Epoch 2 to show different behavior
%     learnnumtrials = 5; %how many "learned" trials you want to compare
% figure;
% 
% for p = 1:learnnumtrials
%     bin1 = min(cell2mat(cellfun(@(x) min(size(x,2)), ROE{1}(end-(learnnumtrials-1):end),'un',0)));%defining the maximum amount of shared bins accross all trials
%     bin2 = min(cell2mat(cellfun(@(x) min(size(x,2)), ROE{2}(end-(learnnumtrials-1):end),'un',0)));
%     
%     subplot(learnnumtrials,1,p) %plotting the distribution of vel in these spatial bins
%     histogram(ROE{1}{end-(learnnumtrials-p)}(1:min(bin1,bin2)),'FaceAlpha',0.5,'Binwidth',0.05) %for epoch 1
%     hold on
%     histogram(ROE{2}{end-(learnnumtrials-p)}(1:min(bin1,bin2)),'FaceAlpha',0.5,'Binwidth',0.05) %for epoch 2
%      xlim([0 1.5])
%     title(['trials ' num2str(size(ROE{1},2)-(learnnumtrials-p)) ' of RewEpoch 1 and ' num2str(size(ROE{2},2)-(learnnumtrials-p)) ' of RewEpoch 2'])
% end
% if eval(version)>= 18 %checks if version of matlab is current enough to use inbed function
%     sgtitle([VR.name_date_vr ' Comparing Speed Distribution of Learned Trials'],'fontsize',14,'interpreter','none')
% else %requires a function mtit
%     mtit([VR.name_date_vr ' Comparing Speed Distribution of Learned Trials'],'fontsize',14,'interpreter','none') %name of the file you are analising as main title
% end
% 
% testROEhist = cell(1,3); %creating a temporary variable for the combined distributions
% 
% for p = 1:2 %for the first two epochs...
%     for j = 1:learnnumtrials %for the number of trials you wish to combine
%         testROEhist{p} = [testROEhist{p} ROE{p}{end-j+1}(1:min(bin1,bin2))];
%         
%     end
%     figure(30)
%      hold on
%     histogram(testROEhist{p},15,'FaceAlpha',0.5)
%     
%     if eval(version)>= 18 %checks if version of matlab is current enough to use inbed function
%         sgtitle([VR.name_date_vr ' Combined Distribution of Last 5 trials for Epoch 1(blue) & Epoch 2(orange)'],'fontsize',14,'interpreter','none')
%     else %requires a function mtit
%         mtit([VR.name_date_vr ' Combined Distribution of Last 5 trials for Epoch 1(blue) & Epoch 2(orange)'],'fontsize',14,'interpreter','none') %name of the file you are analising as main title
%     end
%    figure(31)
%    hold on
%    [f,x] = ecdf(sort(testROEhist{p})'); %create a cumulative distribution
%    cuml{p} = f;
%    cumlx{p} = x;
%    plot(x,f)
%    if eval(version)>= 18 %checks if version of matlab is current enough to use inbed function
%        sgtitle([VR.name_date_vr ' Combined Cumulative Distribution of Last 5 trials for Epoch 1(blue) & Epoch 2(orange)'],'fontsize',14,'interpreter','none')
%    else %requires a function mtit
%        mtit([VR.name_date_vr ' Combined Cumulative Distribution of Last 5 trials for Epoch 1(blue) & Epoch 2(orange)'],'fontsize',14,'interpreter','none') %name of the file you are analising as main title
%    end
%     
% end
% 
% figure(31)
% [h,p] = kstest2( testROEhist{1}',testROEhist{2}'); %Two-sample Kolmogorov-Smirnov test for these distributions
% text(0.5,0.5,['p= ' num2str(p)]);

% %% Sliding Window COM figure
%  
%     figure;
%     %raw data plot
%     subplot(3,8,[1 2 9 10 17 18])
%     if eval(version)>= 18 %checks if version of matlab is current enough to use inbed function
%        sgtitle([VR.name_date_vr ' Sliding Window'],'fontsize',14,'interpreter','none')
%     else %requires a function mtit
%     mtit([VR.name_date_vr ' Sliding Window '],'fontsize',14,'interpreter','none') %name of the file you are analising as main title
%     end
%     scatter(time_min,ypos,1,'.','MarkerEdgeColor',[0.6 0.6 0.6])
%     hold on
%     scatter(islickX,islickYpos,10,'r','filled')
%     scatter(rewX,rewYpos,10,'b','filled')
%     for mm = 1:length(changeRewLoc)-1 %the rectangle indicating the reward location, overlaps the probe trials referring to the previous reward location
%         rectangle('position',[time_min(changeRewLoc(mm)) RewLoc(mm)-rewSize time_min(changeRewLoc(mm+1))-time_min(changeRewLoc(mm)) 2*rewSize],'EdgeColor',[0 0 0 0],'FaceColor',[0 .5 .5 0.3])
%     end
%     ylabel('track position - cm')
%     xlabel('time - minutes')
%     title ('lick (red) along track - and reward (blue)') 
%     
%     for tt = 1:length(RewLoc) % for the three reward locations..
%         if tt<=3 && ~isempty(allCOM{tt})
%             subplot(3,8,[3+8*(tt-1) 4+8*(tt-1)]) % blue normalized lick scatterplot all the trials, succesfull or not are considered
%             hold on
%             line([1,length(allCOM{tt})],[-rewSize,-rewSize],'LineStyle','--','LineWidth',1.5,'Color',	[0 0 0 0.4]) % dashed line is plotted at the nomalized COM (y=0 always) - 7.5
%             for i=1:numel(trialyposlicks{tt}) % the x axis indicates the trial number (eg. if trial 4 has no licks, the location 4 on the x axes will appear empty and the next trial (trial 5) will be plotted at x positon 5)
%                 scatter(ones(size(trialyposlicks{tt}{i})).*i,(trialyposlicks{tt}{i})-RewLoc(tt),38,[0 0.4470 0.7410],'filled')
%             end
%             alpha(.05)
%             scatter((1:length(allCOM{tt})),allCOM{tt},50,'filled')
%             scatter(find(failure{tt}),allCOM{tt}(find(failure{tt})),50,'r','filled') %failed trials are plotted with a red dot.
%             
%             subplot(3,8,[5+8*(tt-1) 6+8*(tt-1)])% black normalized lick scatterplot with std. all the trials, succesfull or not are considered
%             scatter((1:1:length(slidingMeanCOM{tt})),slidingMeanCOM{tt},'k');
%             hold on
%             line([1,length(slidingMeanCOM{tt})],[-rewSize,-rewSize],'LineStyle','--','LineWidth',1.5,'Color',	[0 0 0 0.4])
%             title([ 'reward location ' num2str(tt) ' = ' num2str(RewLoc(tt))])
%             
%             if ~isempty(slidingMeanCOM{tt})
%                 subplot(3,8, [7+8*(tt-1) 8+8*(tt-1)]) % barplot with p value of first and last five trials. only succesfull trials are considered
%             scatter((1:1:length(slidingVarCOM{tt})),slidingVarCOM{tt},'k');
%             hold on
%             line([1,length(slidingMeanCOM{tt})],[-rewSize,-rewSize],'LineStyle','--','LineWidth',1.5,'Color',	[0 0 0 0.4])
%             title([ 'reward location ' num2str(tt) ' = ' num2str(RewLoc(tt))])
%             else
%                 disp('Not enough succesfull trials in the last reward zone to compute sliding window');
%             end
%         end
%     end   
end

% %% InterLick Interval and Total Lick Count
% figure; 
% x = 1:2;
% numtrialcompair = 5;
% for i = 1:size(InterLickInterval,1)
%     successful = find(failure{i} == 0);
% subplot(2,size(InterLickInterval,1),i)
% imagesc(flipud(squeeze(InterLickInterval(i,:,:))'))
% xlabel('trial number')
% y1 = yticklabels;
% yticklabels(flipud(y1))
% ylabel('Interlick Interval Number')
% if length(successful) >= numtrialcompair*2
% earlyInterval = reshape(InterLickInterval(i,successful(1:numtrialcompair),:),1,numtrialcompair*size(InterLickInterval,3));
% lateInterval = reshape(InterLickInterval(i,successful(end-numtrialcompair+1:end),:),1,numtrialcompair*size(InterLickInterval,3));
% earlyInterval(isnan(earlyInterval)) = [];
% lateInterval(isnan(lateInterval)) = [];
% earlymean = nanmean(earlyInterval);
% latemean = nanmean(lateInterval);
% earlystd =nanstd(earlyInterval)/sqrt(length(earlyInterval));
% latestd = nanstd(lateInterval)/sqrt(length(lateInterval));
% subplot(2,size(InterLickInterval,1),i+size(InterLickInterval,1))
% bar(x,[earlymean latemean])
% hold on
% er = errorbar(x,[earlymean...
%     latemean]...
%     ,[earlystd latestd],[earlystd latestd]);
% er.Color = [0 0 0];
% [h,p] = ttest2(earlyInterval,lateInterval);
% text(1.5,max([earlymean latemean] +0.001),['t-test2 p = ' num2str(p)])
%        Labels = {['first ' num2str(numtrialcompair) ' trials'], ['last ' num2str(numtrialcompair) ' trials']};
%                 set(gca,'XTick', 1:2, 'XTickLabel', Labels);
% hold off
% end
% 
% end
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

