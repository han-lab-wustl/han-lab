%% COM analysys multiple trials % single mouse only
clear 
clc

COMbytrial = figure;
variance = figure;
mice_names= [{'e200'} {'e201'}];
days_per_mouse = {[65:76,78:82], [55:73,75:80]};
conditions = {'ep2', 'ep3', 'control'};
srcs = ["Y:\sstcre_imaging", "Z:\sstcre_imaging"];
%iterate through conditions

for condind=1:length(conditions)


meanClose2farByTrial = [];
meanFar2closeByTrial = [];
meanFirstRewEpoch = [];

for mice=1:size(mice_names,2)
    clearvars -except conditions condind mice_names variance COMbytrial mice meanClose2farByTrial meanFar2closeByTrial meanFirstRewEpoch days_per_mouse srcs
    variab = 2;  % set to 2 for reward location absolute distance (1,2,3) , to 3 for relative distance (first rew loc, far to close, close to far)
    
    % files = dir('*).mat');
    mouse_name = mice_names{mice};
    days = days_per_mouse{mice};
    src = srcs(mice);
    files = days; % zd added
    % dumb = struct2cell(files);
    % dumber = (dumb(1,:)');
    % dumbest = cellfun (@(x) x(1:4),dumber,'un',0);
    % [r,c]=find(cell2mat(dumbest)~=(mice_names{mice}));
    % files(r) = [];
    a=1; % counter for rew locs
    b=1;
    c=1;
    
    for i=condind:3:length(files)
            
        %zd added
        daypth = dir(fullfile(src, mouse_name, string(days(i)), "behavior", "vr\*.mat"));
        disp(mouse_name)
        disp(days(i))
        file=fullfile(daypth.folder,daypth.name);
        eval(['load ' file]); %load eac VR structure
        
        rewlocs = unique(VR.changeRewLoc);
        disp(rewlocs(2:end)*(3/2))
        % if VR.scalingFACTOR ==1 % session should be referred to mouse and it has to have more than 2 rew loc
        if sum(VR.changeRewLoc>0)>3 || sum(VR.changeRewLoc>0)==3 && VR.trialNum(end)>15 % changed from 20 to 15 
            % calc the COM
           allLicks = 1; % 1, if you want to use all the licks, 2 if you want to use only licks before reward (reward lick included)
           scalefactor = 1/(2/3); % need to include this in function to characterise rew zone 1,2,3
           [allcom, allstdcom, COM{i}] = COMgeneralviewF(VR, allLicks, scalefactor); %{i} % works fine until here 
           % epoch x trial structure
            for j = 1:size(COM{i},1)
            %      if length(COM{i}{j}) >20 % what is this for?
                 if COM{i}{j,variab}==1 % if assigned to rew lock 1
                   COMsplit{a,COM{i}{j,variab}}=COM{i}{j,1}; % zd added i to preserve epoch, only for saving
                   a=a+1;            
                 elseif COM{i}{j,variab}==2
                   COMsplit{b,COM{i}{j,variab}}=COM{i}{j,1};
                   b=b+1;
                 elseif COM{i}{j,variab}==3
                     COMsplit{c,COM{i}{j,variab}}=COM{i}{j,1};
                     c=c+1;
                 end
            end
        end
    end


%% nan matrix COM
a=1;
nanCOM = cell(1,size(COMsplit,2));

for h = 1:size(COMsplit,2)
            test = COMsplit(:,h);
        [s,d] = cellfun(@size,test);
        out = max([s,d]);
        nanCOM{h} = cell(out);
    for i = 1:size(COMsplit,1)
        if ~isempty(COMsplit{i,h})
            for j = 1:length(COMsplit{i,h})
                nanCOM{h}{j} = [nanCOM{h}{j} COMsplit{i,h}(j)]; 
            end
        end
    end
    meanCOMsplit{1,h} = nan(1,27); %mean
    meanCOMsplit{2,h} = nan(1,27);%standard error
    meanCOMsplit{3,h} = nan(1,27); %variabiance
    for kk = 1:max(out)
    meanCOMsplit{1,h}(kk) = nanmean(nanCOM{h}{kk}); %mean
    meanCOMsplit{2,h}(kk) = nanstd(nanCOM{h}{kk})/sqrt(size(nanCOM{h}{kk},2));%standard error
    meanCOMsplit{3,h}(kk) = var(nanCOM{h}{kk},'omitnan'); %variabiance, corrected by num of obs
    end
end

colors = {'b' 'r' 'k' 'g' 'm' 'c' };
for i = 1:size(COMsplit,2) % zd changed because some mice don't have rew zones yet
    figure(COMbytrial)
    subplot(2,1,mice)% zd changed for her 2 mice
%     subplot(ceil(size(mice_names,2)/2),ceil(size(mice_names,2)/2),mice)
    errorbar(meanCOMsplit{1,i}(),meanCOMsplit{2,i}(),colors{i} ) % set color to {mice} to have same mice
    hold on
end
title(['mean ' mice_names{mice}])
sgtitle('mean of COM split by rew zones')

for i = 1:size(COMsplit,2)
    figure(variance)
    subplot(2,1,mice)% zd changed for her 2 mice
    plot(meanCOMsplit{3,i}(),colors{i})
    hold on
end
title(['variance ' mice_names{mice}])
sgtitle('variance of COM split by rew zones')

for i = 1:1:size(COMsplit,2)
    if i == 3
    meanClose2farByTrial = [meanClose2farByTrial; meanCOMsplit{1,i}(1:22)];
    elseif i == 2
    meanFar2closeByTrial = [meanFar2closeByTrial; meanCOMsplit{1,i}(1:22)];
    elseif i == 1
    meanFirstRewEpoch = [meanFirstRewEpoch; meanCOMsplit{1,i}(1:22)];
    end
end

save(sprintf('Y:\\sstcre_analysis\\hrz\\lick_analysis\\%s_COMsplit_condition_%s', ...
    mouse_name, conditions{condind}), 'COM', 'meanCOMsplit')

end
end

%% final plot 

%color pick@ https://learnui.design/tools/data-color-picker.html#palette
% 
% str = '#505c57';
% color1 = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
% 
% str = '#789c50';
% color2 = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
% 
% str = '#ffc414';
% color3 = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
% 
% x = 1:22;
% N = length(x);
% y1 = nanmean(meanFirstRewEpoch,1);
% y2 = nanmean(meanClose2farByTrial,1);
% y3 = nanmean(meanFar2closeByTrial,1);
% 
% b1 = nanstd(meanFirstRewEpoch,1);
% b2 = nanstd(meanClose2farByTrial,1);
% b3 = nanstd(meanFar2closeByTrial,1);
% 
% 
% ba= figure;
% hold on
% boundedline(x, y1, b1,'cmap', color1, 'alpha');
% boundedline(x, y2, b2,'cmap',  color2, 'alpha');
% boundedline(x, y3, b3,'cmap',  color3, 'alpha');
% hold off
% 
% htitle = title('Lick accuracy');
% set(htitle,'FontSize',14);
% xlabel('Trial number');
% ylabel('Cm to Reward Zone');
% yticks([-50 -25 -7.5]);
% ylim([-59,-7.5])
% xlim([1,22])
% set(gca,'FontSize',12);
% set(gca,'Color', 'none');
% grid on 
% a = legendflex({'First Epoch' 'Far to Close' 'Close to Far'},'buffer', [0 -20],'color','none');

