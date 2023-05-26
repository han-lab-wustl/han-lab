%% COM analysys multiple trials % single mouse only
clear 
clc

COMbytrial = figure;
variance = figure;
mice_names= [{'e200'} {'e201'}];
days_per_mouse = {[65:75],[55:73,75]};
srcs = ["Y:\sstcre_imaging", "Z:\sstcre_imaging"];
meanClose2farByTrial = [];
meanFar2closeByTrial = [];
meanFirstRewEpoch = [];

for mice=1:size(mice_names,2)

    clearvars -except mice_names variance COMbytrial mice meanClose2farByTrial meanFar2closeByTrial meanFirstRewEpoch days_per_mouse srcs
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
 
cOfMass = cell(1, 500);
cOfMassFL = cell(1, 500);

    a=1;
    b=1;
    c=1;
    
for dy=1:length(files)
    
    %zd added
    daypth = dir(fullfile(src, mouse_name, string(days(dy)), "behavior", "vr\*.mat"));
    file=fullfile(daypth.folder,daypth.name);
    eval(['load ' file]); %load eac VR structure

if VR.scalingFACTOR ==1 % session should be referred to mouse and it has to have more than 2 rew loc
if sum(VR.changeRewLoc>0)>3 || sum(VR.changeRewLoc>0)==3 && VR.trialNum(end)>16 % changed from 20 to 16 
    %% calc the COM
   COM{i} = COMgeneralviewF(VR); %{i}
 for j = 1:size(COM{i},1)
     if length(COM{i}{j}) >20
     if COM{i}{j,variab}==1
   COMsplit{a,COM{i}{j,variab}}=COM{i}{j,1};
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
for i = 1:3
    figure(COMbytrial)
    subplot(ceil(size(mice_names,2)/2),ceil(size(mice_names,2)/2),mice)
    errorbar(meanCOMsplit{1,i}(),meanCOMsplit{2,i}(),colors{i} ) % set color to {mice} to have same mice
    hold on
end
title(['mean ' mice_names{mice}])

for i = 1:3
    figure(variance)
    subplot(ceil(size(mice_names,2)/2),ceil(size(mice_names,2)/2),mice)
    plot(meanCOMsplit{3,i}(),colors{i})
    hold on
end
title(['variance ' mice_names{mice}])

for i = 1:3
    if i == 3
    meanClose2farByTrial = [meanClose2farByTrial; meanCOMsplit{1,i}(1:22)];
    elseif i == 2
    meanFar2closeByTrial = [meanFar2closeByTrial; meanCOMsplit{1,i}(1:22)];
    elseif i == 1
    meanFirstRewEpoch = [meanFirstRewEpoch; meanCOMsplit{1,i}(1:22)];
    end
end

end

%% final plot 

%color pick@ https://learnui.design/tools/data-color-picker.html#palette

str = '#505c57';
color1 = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;

str = '#789c50';
color2 = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;

str = '#ffc414';
color3 = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;

x = 1:22;
N = length(x);
y1 = nanmean(meanFirstRewEpoch,1);
y2 = nanmean(meanClose2farByTrial,1);
y3 = nanmean(meanFar2closeByTrial,1);

b1 = nanstd(meanFirstRewEpoch,1);
b2 = nanstd(meanClose2farByTrial,1);
b3 = nanstd(meanFar2closeByTrial,1);


ba= figure
hold on
boundedline(x, y1, b1,'cmap', color1, 'alpha');
boundedline(x, y2, b2,'cmap',  color2, 'alpha');
boundedline(x, y3, b3,'cmap',  color3, 'alpha');
hold off

htitle = title('Lick accuracy');
set(htitle,'FontSize',14);
xlabel('Trial number');
ylabel('Cm to Reward Zone');
yticks([-50 -25 -7.5]);
ylim([-59,-7.5])
xlim([1,22])
set(gca,'FontSize',12);
set(gca,'Color', 'none');
grid on 
a = legendflex({'First Epoch' 'Far to Close' 'Close to Far'},'buffer', [0 -20],'color','none')

