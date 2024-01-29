%% COM analysys multiple trials 

%plot COM by trial with std of different mice.


clear 
clc

mice_names=[{'E144'} {'E146'} {'E139'}];

for mice=1:size(mice_names,2)

variab = 3;  % set to 2 for reward location absolute distance (1,2,3) , to 3 for relative distance (first rew loc, far to close, close to far)

files = dir('*).mat');
dumb = struct2cell(files);
dumber = (dumb(1,:)');
dumbest = cellfun (@(x) x(1:4),dumber,'un',0);
[r,c]=find(cell2mat(dumbest)~=(mice_names{mice}));
files(r) = [];
 
cOfMass = cell(1, 500);
cOfMassFL = cell(1, 500);

    a=1;
    b=1;
    c=1;
    
for i=1:length(files)
    eval(['load ' files(i).name]); %load eac VR structure

if VR.scalingFACTOR ==1 % session should be referred to mouse and it has to have more than 2 rew loc
if sum(VR.changeRewLoc>0)>3 || sum(VR.changeRewLoc>0)==3 && VR.trialNum(end)>20
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
    for kk = 1:max(out)
    meanCOMsplit{mice}{1,h}(kk) = nanmean(nanCOM{h}{kk}); %mean
    meanCOMsplit{mice}{2,h}(kk) = nanstd(nanCOM{h}{kk})/sqrt(size(nanCOM{h}{kk},2));%standard error
    meanCOMsplit{mice}{3,h}(kk) = var(nanCOM{h}{kk},'omitnan'); %variabiance, corrected by num of obs
    end

 miceMean{1,h}(mice,:) = nan(1,27);
miceMean{1,h}(mice,1:size(meanCOMsplit{mice}{1,h},2)) = meanCOMsplit{mice}{1,h}; % cell str is by rew loc, row inside are mice
 miceMean{2,h}(mice,:) = nan(1,27);
miceMean{2,h}(mice,1:size(meanCOMsplit{mice}{3,h},2)) = meanCOMsplit{mice}{3,h};
end
end


figure;
for i = 1:3
    errorbar(nanmean(miceMean{1,i},1),nanstd(miceMean{1,i},1))
    hold on
end
title('mean')

figure;
for i = 1:3
    errorbar(nanmean(miceMean{2,i},1),nanstd(miceMean{2,i},1))
    hold on
end
title('variance')

