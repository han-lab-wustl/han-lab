% Zahra
% analyze behavior of mouse in HRZ memory consolidation experiment
% look at fraction of licks in normal vs. probe trials
% https://www.nature.com/articles/s41593-022-01050-4
close all; clear all;

% select files
[filename,filepath] = uigetfile('*.mat','MultiSelect','on');

ind = 1;
days = filename;
grayColor = [.7 .7 .7];
for dy=1:length(days)
    mouse_pth = fullfile(filepath,filename{dy});
    mouse = load(mouse_pth);    
    % get success and fail trials
    [s,f,str, ftr, ttr, tr] = get_success_failure_trials(mouse.VR.trialNum, mouse.VR.reward);
    success_prop{ind} = s/tr;
    fail_prop{ind} = f/tr;    
    eps = find(mouse.VR.changeRewLoc>0);
    eps = [eps length(mouse.VR.changeRewLoc)]; 
    gainf = 1/mouse.VR.scalingFACTOR;
    rewloc = mouse.VR.changeRewLoc(mouse.VR.changeRewLoc>0)*gainf;
    rewsize = mouse.VR.settings.rewardZone*gainf;
    fig = figure('Renderer', 'painters');
    ypos = mouse.VR.ypos*(gainf);
    velocity = mouse.VR.ROE(2:end)*-0.013./diff(mouse.VR.time);
    scatter(1:length(ypos), ypos, 2, 'filled', 'MarkerFaceColor', grayColor); hold on; 
    plot(find(mouse.VR.lick),ypos(find(mouse.VR.lick)), ...
        'r.', 'MarkerSize',5) 
    for mm = 1:length(eps)-1 %the rectangle indicating the reward location, overlaps the probe trials referring to the previous reward location
        rectangle('position',[eps(mm) rewloc(mm)-rewsize/2 ...
            eps(mm+1)-eps(mm) rewsize],'EdgeColor',[0 0 0 0],'FaceColor',[0 .5 .5 0.3])
    end
    plot(find(mouse.VR.reward==0.5),ypos(find(mouse.VR.reward==0.5)),'b*', ...
        'MarkerSize',5)
    ylabel("Track Position (cm)")    
    xticks([0:10000:length(mouse.VR.time)])
    tic = floor(mouse.VR.time(1:10000:end)/60);
    xticklabels(tic)
    xlabel("Time (minutes)")
    ylim([0 270])
    xlim([0 length(mouse.VR.reward)])
    yticks([0:90:270])
    yticklabels([0:90:270])
    sgtitle(string(filename{dy}))
%     legend({'Position', 'Licks', 'Conditioned Stimulus'})
%     saveas(fig, 'C:\Users\Han\Box\neuro_phd_stuff\han_2023\dlc\dlc_poster_2023\behavior.svg')
%     close(fig)
    ind=ind+1;    
end
% 
% plot trial performance as bar graph
% x = [success_prop{:}];
% y = [fail_prop{:}];
% figure;
% bar([mean(y);mean(x)]','grouped','FaceColor','flat');
% hold on
% plot(1,y,'ok')
% plot(2,x,'ok')
% xticklabels(["Fails" "Successes"])
% ylabel("Proportion of trials")
