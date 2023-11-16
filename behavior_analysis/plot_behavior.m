% Zahra
% analyze behavior of mouse in HRZ
% look at fraction of licks in normal vs. probe trials
% https://www.nature.com/articles/s41593-022-01050-4
close all;
mouse_name = "e218";
days = [20:24];
src = "X:\vipcre";
grayColor = [.7 .7 .7];
ind = 1;
for day=days
    daypth = dir(fullfile(src, mouse_name, string(day), "behavior", "vr\*.mat"));    
    mouse = load(fullfile(daypth.folder,daypth.name));    
    % get success and fail trials
    [s,f,tr] = get_success_failure_trials(mouse.VR.trialNum, mouse.VR.reward);
    success_prop{ind} = s/tr;
    fail_prop{ind} = f/tr;
    disp(day); disp(mouse.VR.lickThreshold)
    figure;
    ypos = mouse.VR.ypos*(3/2);
    velocity = mouse.VR.ROE(2:end)*-0.013./diff(mouse.VR.time);
    plot(ypos, 'Marker', '.', 'MarkerSize',2, 'Color', grayColor); hold on; 
    plot(mouse.VR.changeRewLoc*(3/2), 'k', 'LineWidth',3)
    plot(find(mouse.VR.lick),ypos(find(mouse.VR.lick)), ...
        'r.', 'MarkerSize',5) 
%     plot(find(mouse.VR.reward==0.5),ypos(find(mouse.VR.reward==0.5)),'b*', ...
%         'MarkerSize',10)
    plot(find(mouse.VR.reward==1),ypos(find(mouse.VR.reward==1)),'b*', ...
        'MarkerSize',10)
    ylabel("track length (cm)")
    xlabel("frames")
    title(sprintf('day %i', day))    
    ind=ind+1;
end

% plot trial performance as bar graph
x = [success_prop{:}];
y = [fail_prop{:}];
figure;
bar([mean(y);mean(x)]','grouped','FaceColor','flat');
hold on
plot(1,y,'ok')
plot(2,x,'ok')
xticklabels(["Fails" "Successes"])
ylabel("Proportion of trials")
