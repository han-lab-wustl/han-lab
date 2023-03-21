clc
clear %clean up the environment
[filename,filepath] = uigetfile('*.mat','MultiSelect','on');
file = [filepath filename];
load(file) %load it

cd (filepath); %EH set path

COMgeneralanalysis

figure;
scatter(time_min,ypos,1,'.','MarkerEdgeColor',[0.6 0.6 0.6])
hold on
scatter(islickX,islickYpos,10,'r','filled')
scatter(rewX,rewYpos,10,'b','filled')
for mm = 1:length(changeRewLoc)-1 %the rectangle indicating the reward location, overlaps the probe trials referring to the previous reward location
    rectangle('position',[time_min(changeRewLoc(mm)) RewLoc(mm)-rewSize time_min(changeRewLoc(mm+1))-time_min(changeRewLoc(mm)) 2*rewSize],'EdgeColor',[0 0 0 0],'FaceColor',[0 .5 .5 0.3])
end
ylabel('track position - cm')
xlabel('time - minutes')