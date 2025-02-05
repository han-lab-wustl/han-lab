
clear all; close all
load("Z:\VIP_intswanalysisv9.mat")
%%
close all
dff_probes = {};
for dy=1:5
    fall = VIP_ints(1).NR.day{1,dy}.all;
    plns = fall.Falls;
    planes = 3;
    bin_size = 0.2; %s
    range = 8; %s
    changeRewLoc = fall.rewardLocation{1}(:,1)';
    rewards = fall.rewards{1}(:,1)';
    timedFF = fall.time{1}(:,1)';
    forwardvel = fall.Forwards{1}(:,1)';
    licks = fall.licks{1}(:,1)';
    eps = find(changeRewLoc>0);
    eps = [eps length(changeRewLoc)];
    track_length = 180; %cm; TODO: import from VR instead
    nbins = track_length/bin_size;
    rewlocs = changeRewLoc(changeRewLoc>0);
    grayColor = [.7 .7 .7];

    for pln=1:planes
        dff = fall.Falls{pln}; dff = dff; % invert for gm struct
        trialnum = fall.trialNum{1}(:,1)';

        for ep=2:length(eps)-1
            if length(eps(ep):eps(ep+1))>500
            dff_ep = dff(eps(ep):eps(ep+1),:);
            licks_ep = licks(eps(ep):eps(ep+1));
            rewards_ep = rewards(eps(ep):eps(ep+1));
            ypos = fall.ybinned{1}(:,1); ypos = ypos(eps(ep):eps(ep+1));
            rewloc = rewlocs(ep-1);
            rewloc_mask = zeros(size(rewards_ep));
            rewloc_mask((ypos>rewloc-10) & (ypos<rewloc-8))=1;
            stretches = findConsecutiveNonZeroIndices(rewloc_mask);
            startofrew = zeros(size(rewards_ep)); % compare to previous ep
            startofrew(stretches(:,1))=1;
            % compare to new rew zone (ctrl)
            rewloc = rewlocs(ep);
            rewloc_mask = zeros(size(rewards_ep));
            rewloc_mask((ypos>rewloc-10) & (ypos<rewloc+10))=1;
            stretches = findConsecutiveNonZeroIndices(rewloc_mask);
            startofrew_ctrl= zeros(size(rewards_ep)); % compare to previous ep
            startofrew_ctrl(stretches(:,1))=1;

            trialnum_ep = trialnum(eps(ep):eps(ep+1));
            % for ii=1:size(dff_ep,2)
            %     fig = figure('Renderer', 'painters');
            %     plot(ypos(trialnum_ep<10), 'LineWidth', 2, 'Color', grayColor); hold on;
            %     plot(find(licks_ep(trialnum_ep<10)),ypos(find(licks_ep(trialnum_ep<10))), ...
            %         'k.', 'MarkerSize',7)
            %     plot(find(rewloc_mask(trialnum_ep<10)),ypos(find(rewloc_mask(trialnum_ep<10))), ...
            %         'b.', 'MarkerSize',20)
            %     yyaxis right
            %     plot(dff_ep((trialnum_ep<10),ii))
            % end
            probetotest = 3;
            time_ep = timedFF(eps(ep):eps(ep+1));
            [binnedPerireward_probes,allbins,rewdFF,normmeanrewdFF] = perirewardbinnedactivity(dff_ep(trialnum_ep<probetotest,:), ...
                startofrew(trialnum_ep<probetotest), ...
                time_ep(trialnum_ep<probetotest), range,bin_size);
            fig = figure('Renderer', 'painters');
            subplot(1,2,1)
            imagesc(normalize(binnedPerireward_probes))
            ylabel('Cells')
            xticks(0:25:size(allbins,2))
            xticklabels(-range:5:range)
            % correct trials
            [binnedPerireward_success,allbins,rewdFF,normmeanrewdFF] = perirewardbinnedactivity(dff_ep(trialnum_ep>3,:),rewards_ep(trialnum_ep>3), ...
                time_ep(trialnum_ep>3), range,bin_size);
            subplot(1,2,2)
            imagesc(normalize(binnedPerireward_success))
            ylabel('Cells')
            xticks(0:25:size(allbins,2))
            xticklabels(-range:5:range)
            % control
            [binnedPerireward_probes_ctrl,allbins,rewdFF,normmeanrewdFF] = perirewardbinnedactivity(dff_ep(trialnum_ep<probetotest,:), ...
                startofrew_ctrl(trialnum_ep<probetotest), ...
                time_ep(trialnum_ep<probetotest), range,bin_size);

            % get only probes
            dff_probes{dy,pln,ep} = [binnedPerireward_probes,binnedPerireward_probes_ctrl,binnedPerireward_success];
            end
        end
    end
end
%%
save('Z:\vip_dff_probes_3probe.mat','dff_probes')
