function plotVR
load('C:\Users\imaging_VR\Documents\MATLAB\tillTheEnd_variables\summary','name_date')  
figure; title('reward locationd change in the past three experiments')
    for i=1:3
    subplot(1,3,i)
    load(['C:\Users\imaging_VR\Documents\MATLAB\tillTheEnd_variables\VR_data\' name_date{(size(name_date,1)+1)-(i),1} '.mat'],'VR')
    file=VR.changeRewLoc;
    plot(file(file>1))
    ylim([40 160]);
    if i==2
    title('reward location in the past three experiments (most recent to oldest)')
    end
    end
end