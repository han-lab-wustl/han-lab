% zahra's opto analysis for vip
clear all; close all
mouse_name = "e218";
days = [35,36,37,38,39,40];
cells_to_plot = {[453,63,26,38]+1, [111,41,65,2]+1, [72,41,27,14]+1,...
    [301 17 13 320]+1, [98 33 17 3]+1, [92 20 17 26]+1}; % indices of red cells from suite2p per day
src = "X:\vipcre";
dffs_cp_dys = {};
dyind = 1;
for dy=days
    daypth = dir(fullfile(src, mouse_name, string(dy), "**\*Fall.mat"));
    load(fullfile(daypth.folder,daypth.name));
    % plot with behavior with red cells
    %%
    grayColor = [.7 .7 .7];
    eps = find(changeRewLoc>0);
    eps = [eps length(changeRewLoc)];
    gainf = 1/VR.scalingFACTOR;
    rewloc = changeRewLoc(changeRewLoc>0)*gainf;
    rewsize = VR.settings.rewardZone*gainf;
    ypos = ybinned*(gainf);
    velocity = forwardvel;
    for cp=cells_to_plot{dyind}
        fig = figure('Renderer', 'painters');
        scatter(1:length(ypos), ypos, 2, 'filled', 'MarkerFaceColor', grayColor); hold on;
        plot(find(licks),ypos(find(licks)), ...
            'r.', 'MarkerSize',5)
        for mm = 1:length(eps)-1 %the rectangle indicating the reward location, overlaps the probe trials referring to the previous reward location
            rectangle('position',[eps(mm) rewloc(mm)-rewsize/2 ...
                eps(mm+1)-eps(mm) rewsize],'EdgeColor',[0 0 0 0],'FaceColor',[0 .5 .5 0.3])
        end
        plot(find(rewards==0.5),ypos(find(rewards==0.5)),'b*', ...
            'MarkerSize',5)
        ylabel("Track Position (cm)")
        xticks([0:10000:length(timedFF)])
        tic = floor(timedFF(1:10000:end)/60);
        ticnan = ones(size(tic))*NaN;
        ticnan(1) = tic(1);
        ticnan(end) = tic(end);
        xticklabels(ticnan)
        xlabel("Time (minutes)")
        ylim([0 270])
        xlim([0 length(rewards)])
        yticks([0:90:270])
        yticklabels([0:90:270])
        yyaxis right
        plot((dFF(:,cp)+1), 'k')
        ylim([0.5 4])
    end
    %%
    dffs_cp = zeros(length(cells_to_plot{dyind}), length(eps)-1);
    cpind = 1;
    for cp=cells_to_plot{dyind}
        dffs = [];
        % pre reward activity led on vs. led off
        for ep=1:length(eps)-1
            eprng = eps(ep):eps(ep+1);
            ypos = ybinned(eprng);
            rewloc_ = rewloc(ep);
            iind = ypos<(rewloc_-5); % rewsize 5 cm
            dff_ep = dFF(eprng,cp);
            dff_ep_ = dff_ep(iind);
            zscore_dff = (dff_ep_-mean(dff_ep, 'omitnan'))/std(dff_ep, 'omitnan');
            dffs(ep) = mean(zscore_dff, 'omitnan');
        end
        dffs_cp(cpind,:) = dffs;
        cpind = cpind+1;
    end
    dffs_cp_dys{dyind} = dffs_cp; % collect per day
    dyind = dyind+1;
end