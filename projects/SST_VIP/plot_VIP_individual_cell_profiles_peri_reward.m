% Zahra adaptation of gm's code
% VIP individual cell profiles
clear all; close all
load("Z:\VIP_intswanalysisv9.mat")
an = 'vip';
%%
dy = 2; % day
fall = VIP_ints(1).NR.day{1,dy}.all;
plns = fall.Falls;
planes = 3;
bin_size = 0.2; %s
range = 5; %s
changeRewLoc = fall.rewardLocation{1}(:,1)';
rewards = fall.rewards{1}(:,1)';
timedFF = fall.time{1}(:,1)';
forwardvel = fall.Forwards{1}(:,1)';
eps = find(changeRewLoc>0);
eps = [eps length(changeRewLoc)];
track_length = 180; %cm; TODO: import from VR instead
nbins = track_length/bin_size;
rewlocs = changeRewLoc(changeRewLoc>0);
grayColor = [.7 .7 .7]; purple = [0.4940, 0.1840, 0.5560];
savedst = 'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\figure_data\vip';
%%
for pln = 1:planes
    dff = fall.Falls{pln}; dff = dff; % invert for gm struct
    [binnedPerireward, allbins, rewdFF, normmeanrewdFF] = perirewardbinnedactivity(dff, rewards, timedFF, ...
        range, bin_size);
    [binnedvel, ~, rewvel] = perirewardbinnedvelocity(forwardvel, rewards, timedFF, ...
        range, bin_size);

    for cll = 1:size(dff, 2)
        fig = figure('Renderer', 'painters');%, 'WindowState', 'maximized');
        for rewind = 1:size(rewdFF, 3)
            try % to allow for velocity plot
                plot(rewdFF(:, cll, rewind), 'Color', grayColor); hold on % plot each trial
            catch
            end
        end
        try
            plot(binnedPerireward(cll, :), 'k', 'LineWidth', 1.5); hold on % mean trial plot each cell
        catch
        end
        xline(median(1:size(allbins, 2)), 'b--', 'LineWidth', 1.5) % mark reward
        xticks(0:25:size(allbins, 2))
        xticklabels(-range:5:range)
        xlabel('Time from Reward (s)')
        ylabel('dF/F')

        sgtitle(sprintf('Successful trials, plane %i, cell %i', pln, cll))
        % export_fig(fullfile(savedst, sprintf('%s_successful_trials_cell_profile_peri_reward_plane%i_cell%i', an, pln, cll)), '-svg')
        % close(fig)
    end
end
