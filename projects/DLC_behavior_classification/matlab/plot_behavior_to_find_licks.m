clear all; close all;
[filename,filepath] = uigetfile('*.mat','MultiSelect','on');
grayColor = [.7 .7 .7];

for fl=1:numel(filename)
    flnm = fullfile(filepath, filename{fl});
    mouse=load(flnm);   
    figure;
    subplot(3,1,1)
    plot(mouse.VR.imageSync, 'g'); alpha(.2); hold on; 
    ylabel("imagesync")
    subplot(3,1,2)
    plot(mouse.VR.lickVoltage, 'r'); ylim([-0.15 0.05]);
    ylabel("licks")
    subplot(3,1,3)
    forwardvel = -0.013*mouse.VR.ROE(2:end)./diff(mouse.VR.time);    
    plot(forwardvel, 'Color', grayColor)
    ylabel("velocity")
    ylim([-100 300])
    xlabel("frames")

    sgtitle(sprintf("%s", flnm))
    figure;
    plot(mouse.VR.imageSync, 'g'); alpha(.5); hold on; 
    plot(mouse.VR.lickVoltage, 'r'); ylim([-0.15 0.05]);
%     plot(forwardvel/300, 'Color', grayColor)
%     ylim([-100 300])
    xlabel("frames")
    title(sprintf("%s", flnm))
end