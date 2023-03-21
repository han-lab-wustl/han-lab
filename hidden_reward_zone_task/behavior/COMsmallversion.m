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
% [file name,filepath] = uigetfile('*.mat','MultiSelect','on'); % pick the VR file you are interested in
filename= {'E138_06_Nov_2020_time(13_03_39).mat','E138_04_Nov_2020_time(12_58_17).mat','E138_02_Nov_2020_time(13_11_30).mat','E138_01_Nov_2020_time(12_57_50).mat','E138_30_Oct_2020_time(16_11_10).mat'};
filepath = 'N:\VR_data\';
hs = [];
hsCOM = [];
hsRatio = [];
hsLickDistance = [];
hsmeanROE = [];
hsmeancutROE = [];
RewLocs = [];
alltests = {'hs','hsCOM','hsRatio','hsLickDistance','hsmeanROE','hsmeancutROE'};

for days = 1:numel(filename)
clc
clearvars -except filename filepath days figure(100000) alltests RewLocs hs hsCOM hsRatio hsLickDistance hsmeanROE hsmeancutROE
version = ver;
version = version.Release;
version = version(5:6);

%EH variables
rewWinNeg=10;% EH -cm from rewStart peri-reward window
rewWinPos=5;% EH +cm peri-reward window
rewSize=7.5; %EH 15cm rew zone + -
numTrialsStart=5; %EH number of trials to average at start and end of rew epoch
numTrialsEnd=5; %EH number of trials to average at start and end of rew epoch

% GM Variables
speedbinsize = 3;
conversion = -0.013;
slidingwindow = 5;
% 
% 

file = [filepath filename{days}];
load(file) %load it
if length(find(VR.changeRewLoc)) > 1
cd (filepath); %EH set path
%%
COMgeneralanalysis
    

%% InterLick Interval and Total Lick Count
% figure;
% x = 1:2;
% numtrialcompair = 3;
% for i = 1:size(InterLickInterval,1)
%     successful = find(failure{i} == 0);
%     subplot(2,size(InterLickInterval,1),i)
%     imagesc(flipud(squeeze(InterLickInterval(i,:,:))'))
%     xlabel('trial number')
%     y1 = yticklabels;
%     yticklabels(flipud(y1))
%     ylabel('Interlick Interval Number')
%     if length(successful) >= numtrialcompair*2
%         earlyInterval = reshape(InterLickInterval(i,successful(1:numtrialcompair),:),1,numtrialcompair*size(InterLickInterval,3));
%         lateInterval = reshape(InterLickInterval(i,successful(end-numtrialcompair+1:end):length(failure{i}),:),1,numtrialcompair*size(InterLickInterval,3));
%         earlyInterval(isnan(earlyInterval)) = [];
%         lateInterval(isnan(lateInterval)) = [];
%         earlymean = nanmean(earlyInterval);
%         latemean = nanmean(lateInterval);
%         earlystd =nanstd(earlyInterval)/sqrt(length(earlyInterval));
%         latestd = nanstd(lateInterval)/sqrt(length(lateInterval));
%         subplot(2,size(InterLickInterval,1),i+size(InterLickInterval,1))
%         bar(x,[earlymean latemean])
%         hold on
%         er = errorbar(x,[earlymean...
%             latemean]...
%             ,[earlystd latestd],[earlystd latestd]);
%         er.Color = [0 0 0];
%         [h,p] = ttest2(earlyInterval,lateInterval);
%         text(1.5,max([earlymean latemean] +0.001),['t-test2 p = ' num2str(p)])
%         Labels = {['first ' num2str(numtrialcompair) ' trials'], ['last ' num2str(numtrialcompair) ' trials']};
%         set(gca,'XTick', 1:2, 'XTickLabel', Labels);
%         hold off
%     end
%     
% end
    figure(1000000)
    subplot(ceil(numel(filename)/4),min(4,numel(filename)),days)
    x = 1:2;
numtrialcompair = 5;
ticks = 0;
for i = 1:size(InterLickInterval,1)
    x = i*2-1:i*2;
    successful = find(failure{i} == 0);
    
    if length(successful) >= numtrialcompair*2
        ticks = ticks+1;
        earlyInterval = reshape(InterLickInterval(i,successful(1:numtrialcompair),:),1,numtrialcompair*size(InterLickInterval,3));
        lateInterval = reshape(InterLickInterval(i,successful(end-numtrialcompair+1:end),:),1,numtrialcompair*size(InterLickInterval,3));
        earlyInterval(isnan(earlyInterval)) = [];
        lateInterval(isnan(lateInterval)) = [];
        earlymean = nanmean(earlyInterval);
        latemean = nanmean(lateInterval);
        earlystd =nanstd(earlyInterval)/sqrt(length(earlyInterval));
        latestd = nanstd(lateInterval)/sqrt(length(lateInterval));
        b = bar(x,[earlymean latemean]);
        b.FaceColor = 'flat';
        b.CData(1,:) = [137 207 240]/256;
        b.CData(2,:) = [234 60 83]/256;
        hold on
        er = errorbar(x,[earlymean...
            latemean]...
            ,[earlystd latestd],[earlystd latestd]);
        er.Color = [0 0 0];
        [h,p] = ttest2(earlyInterval,lateInterval);
        hs = [hs h];
        if h == 1
        text(mean(x),max([earlymean latemean] +0.001),'*','FontSize',16,'FontWeight','bold')
        end
    end
    
end 
xticks(1.5:2:2*ticks-0.5)
xticklabels(1:ticks)
xlabel('Epoch')
ylabel('Average Interlick Interval')
title(VR.name_date_vr,'fontsize',14,'interpreter','none')

    figure(1000001)
    subplot(ceil(numel(filename)/4),min(4,numel(filename)),days)
    x = 1:2;
ticks = 0;
for i = 1:size(InterLickInterval,1)
    x = i*2-1:i*2;
    successful = find(failure{i} == 0);
    
    if length(successful) >= numtrialcompair*2
        ticks = ticks+1;
        earlyInterval = reshape(InterLickInterval(i,successful(1:numtrialcompair),:),1,numtrialcompair*size(InterLickInterval,3));
        lateInterval = reshape(InterLickInterval(i,successful(end-numtrialcompair+1:end),:),1,numtrialcompair*size(InterLickInterval,3));
        earlyInterval(isnan(earlyInterval)) = [];
        lateInterval(isnan(lateInterval)) = [];
        earlymean = length(earlyInterval)+1;
        latemean = length(lateInterval)+1;
%         earlystd =nanstd(earlyInterval)/sqrt(length(earlyInterval));
%         latestd = nanstd(lateInterval)/sqrt(length(lateInterval));
        b = bar(x,[earlymean latemean]);
        b.FaceColor = 'flat';
        b.CData(1,:) = [137 207 240]/256;
        b.CData(2,:) = [234 60 83]/256;
        hold on
%         er = errorbar(x,[earlymean...
%             latemean]...
%             ,[earlystd latestd],[earlystd latestd]);
%         er.Color = [0 0 0];
%         [h,p] = ttest2(earlyInterval,lateInterval);
%         hs = [hs h];
%         if h == 1
%         text(mean(x),max([earlymean latemean] +0.001),'*','FontSize',16,'FontWeight','bold')
%         end
    end
    
end 
xticks(1.5:2:2*ticks-0.5)
xticklabels(1:ticks)
xlabel('Epoch')
ylabel('Total number of Licks')
title(VR.name_date_vr,'fontsize',14,'interpreter','none')

%check every kind of h done before
for tt = 1:length(RewLoc)
    if ~isempty(allCOM{tt})
        if sum(abs(allCOM{tt})>0)>10 && ~isempty(COM{tt})

             succ = find(abs(COM{tt})>0);
             if length(succ) >= (numTrialsStart+numTrialsEnd)
             RewLocs = [RewLocs RewLoc(tt)];
             [h,p1]=ttest2(COM{tt}(succ(1:numTrialsStart)),COM{tt}(succ(end-(numTrialsEnd-1):end))); %EH perform unpaired t-test
             hsCOM = [hsCOM h];
             [h,p1]=ttest2((allRatio{tt}(succ(1:numTrialsStart))),(allRatio{tt}(succ(end-(numTrialsEnd-1):end)))); %perform t-test. %EH
             hsRatio = [hsRatio h];
                lickDistTrim=[];
                keepLickDist=[];
                earlyLickDist=[];
                lateLickDist=[];
                lickDistTrim=lickDist(tt,:);
                keepLickDist = any(~cellfun('isempty',lickDistTrim), 1);  %// keep columns that don't only contain []
                lickDistTrim = lickDistTrim(:,keepLickDist);
                earlyLickDist=cat(2,lickDistTrim{1,1:numTrialsStart});
                lateLickDist=cat(2,lickDistTrim {1,end-(numTrialsEnd-1):end});
                [h,p1]=ttest2(earlyLickDist,lateLickDist); %perform t-test
                hsLickDistance = [hsLickDistance h];
                [h,p1]=ttest2(meanROE{tt}(succ(1:numTrialsStart)),meanROE{tt}(succ(end-(numTrialsEnd-1):end)));
                hsmeanROE = [hsmeanROE h];
                [h,p1]=ttest2(meancutROE{tt}(succ(1:numTrialsStart)),meancutROE{tt}(succ(end-(numTrialsEnd-1):end)));
                hsmeancutROE = [hsmeancutROE h];
             end
        end
    end
end
end
end
figure(1000000)
    if eval(version)>= 18 %checks if version of matlab is current enough to use inbed function
       sgtitle(' InterLick Interval','fontsize',14,'interpreter','none')
    else %requires a function mtit
    mtit(' InterLick Interval','fontsize',14,'interpreter','none') %name of the file you are analising as main title
    end
    figure(1000001)
    if eval(version)>= 18 %checks if version of matlab is current enough to use inbed function
       sgtitle(' Total Lick Count','fontsize',14,'interpreter','none')
    else %requires a function mtit
    mtit(' Total Lick Count','fontsize',14,'interpreter','none') %name of the file you are analising as main title
    end
    figure;
bar(1:2, [length(find(hs))/length(hs) 1-length(find(hs))/length(hs)])
xticklabels({'Rejected Null','Did not Reject'})
ylabel('Percentage of Epochs')
title('Ttest Overall Results')

figure;

bar(1:2, [length(find(hs))/length(hs) 1-length(find(hs))/length(hs)])
hold on
bar(3:4, [length(find(hsCOM))/length(hsCOM) 1-length(find(hsCOM))/length(hsCOM)])

bar(5:6, [length(find(hsRatio))/length(hsRatio) 1-length(find(hsRatio))/length(hsRatio)])

bar(7:8, [length(find(hsLickDistance))/length(hsLickDistance) 1-length(find(hsLickDistance))/length(hsLickDistance)])

bar(9:10, [length(find(hsmeanROE))/length(hsmeanROE) 1-length(find(hsmeanROE))/length(hsmeanROE)])

bar(11:12, [length(find(hsmeancutROE))/length(hsmeancutROE) 1-length(find(hsmeancutROE))/length(hsmeancutROE)])
xticks((1.5:2:11.5))
testlabels = {'ILI','COM','Ratio','LickDistance','ROE','CutROE'};
xticklabels({'ILI','COM','Ratio','LickDistance','ROE','CutROE'})
ylabel('Percentage of Epochs')
title('Ttest Overall Results all tests')

figure;
for t = 1:numel(alltests)
    subplot(2,ceil(numel(alltests)/2),t)
    currenttest = eval(alltests{t});
    scatter(RewLocs,currenttest)
    title(testlabels{t})
    xlabel('RewardPosition')
    ylabel('h values')
    
end
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

