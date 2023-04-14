% Zahra
% analyze behavior of mouse in HRZ
% look at fraction of licks in normal vs. probe trials
% https://www.nature.com/articles/s41593-022-01050-4

mouse_name = "e201";
days = [26:39];
src = "Z:\sstcre_imaging";
grayColor = [.7 .7 .7];

for day=days
    daypth = dir(fullfile(src, mouse_name, string(day), "behavior", "vr\*.mat"));    
    mouse = load(fullfile(daypth.folder,daypth.name));
    disp(mouse.VR.lickThreshold)
    figure;
    velocity = mouse.VR.ROE(2:end)*-0.013./diff(mouse.VR.time);
    plot(mouse.VR.ypos, 'Color', grayColor); hold on; 
    plot(mouse.VR.changeRewLoc, 'b')
    plot(find(mouse.VR.lick),mouse.VR.ypos(find(mouse.VR.lick)),'r.') 
    ylabel("track length (cm)")
    xlabel("frames")
    title(sprintf('day %i', day))        
end