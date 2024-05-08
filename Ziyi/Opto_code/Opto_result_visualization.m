% Clear all workspace variables and close all figures
clear all
close all

% User selects directories through a GUI, which are stored as paths
pr_dir = uipickfiles; 

% Create a list of days to process
days_check = 1:length(pr_dir);

% Define color schemes for different imaging planes
planecolors = {[0 0 1], [0 1 0], [204 164 61]/256, [231 84 128]/256};

% Define a time window for analysis, in seconds
timewindow = 10; 

% Loop through each day selected
for days = days_check
    % Find directories containing Suite2P output for all planes
    dir_s2p = struct2cell(dir([pr_dir{days} '\**\suite2p']));
    planefolders = dir_s2p(:,~cellfun(@isempty,regexp(dir_s2p(1,:),'plane')));

    % Loop through each plane folder found
    for allplanes = 1:size(planefolders,2) 
        % Clear variables except those needed to avoid memory issues
        clearvars -except mouse_id pr_dir days_check days dir_s2p planefolders allplanes planecolors timewindow

        % Construct path to the processed imaging data for the current plane
        pr_dir2 = strcat(planefolders{2, allplanes}, '\plane', num2str(allplanes-1), '\reg_tif\');

        % Change to the directory containing the data
        cd(pr_dir2)

        % Load parameters and calculate baseline-normalized delta F/F
        load('params')
        dFF = params.roibasemean3{1};
        dFF_base_mean = mean(dFF);
        dFF = dFF / dFF_base_mean;

        % Create a new figure for plotting raw data and normalization for the current day
        find_figure(['Raw Data Day ' num2str(days)])
        plot(timedFF, (dFF-min(dFF))/range(dFF)+allplanes-1, 'Color', planecolors{allplanes})
        hold on
        xlims = xlim;
        plot([0.9*xlims(2), 0.9*xlims(2)], [allplanes-0.02/range(dFF), allplanes], 'k-', 'LineWidth',1.5)

        % Detect and label periods of stimulation
        abfstims = bwlabel(stims > 0.5);
        for dw = 1:max(abfstims)-1
            if utimedFF(find(abfstims == dw+1, 1)) - utimedFF(find(abfstims == dw, 1, 'last')) < 0.5
                abfstims(find(abfstims == dw, 1):find(abfstims == dw+1, 1)) = dw+1;
            end
        end
        abfstims = abfstims > 0.5;
        abfrect = consecutive_stretch(find(abfstims));
        abfrect(cellfun(@length, abfrect) < 50) = [];

        % Plot velocity data if this is the last plane
        if allplanes == size(planefolders,2)
            plot(utimedFF, rescale(forwardvelALL, -1, 0), 'k-')
            ylims = ylim;
            for r = 1:length(abfrect)
                rectangle('Position', [utimedFF(abfrect{r}(1)), ylims(1), utimedFF(length(abfrect{r})), ylims(2) - ylims(1)], 'FaceColor', [0 0.5 0.5 0.3], 'EdgeColor', [0 0 0 0])
            end
        end

        % Generate plots for peri-conditioned optogenetic stimulus events
        find_figure(['PeriCSopto Start Day ' num2str(days)])
        subplot(2,4,allplanes*2-1)
        hold on
        xs = cellfun(@(x) utimedFF(x(find(solenoid2ALL(x),1))), rewrect, 'UniformOutput',1);
        for x = 1:length(xs)
            currx = find(timedFF>=(xs(x)-timewindow)&timedFF<(xs(x)+timewindow));
            yyaxis left
            plot(timedFF(currx)-xs(x),dFF(currx),'-','Color',planecolors{allplanes})
            ylim([0.85 1.2])
            hold on
            yyaxis right
            currspeedx = find(utimedFF>=(xs(x)-timewindow)&utimedFF<(xs(x)+timewindow));
            plot(utimedFF(currspeedx)-xs(x),forwardvelALL(currspeedx),'k-')
            ylim([-5 200])
            yticks([0:25:75])
        end
        title(['Peri CS Opto Plane ' num2str(allplanes)])

        % Generate plots for random non-optogenetic control events
        subplot(2,4,allplanes*2)
        for x = 1:length(xs)
            randCS = randi(length(nonCS));
            randxs = nonCS(randCS);
            nonCS(randCS) = [];
            currx = find(timedFF>=(randxs-timewindow)&timedFF<(randxs+timewindow));
            yyaxis left
            plot(timedFF(currx)-randxs,dFF(currx),'-','Color',planecolors{allplanes})
            ylim([0.85 1.2])
            hold on
            yyaxis right
            currspeedx = find(utimedFF>=(randxs-timewindow)&utimedFF<(randxs+timewindow));
            plot(utimedFF(currspeedx)-randxs,forwardvelALL(currspeedx),'k-')
            ylim([-5 200])
            yticks([0:25:75])
        end
        title(['Peri CS nonOpto Plane ' num2str(allplanes)])
    end
end
