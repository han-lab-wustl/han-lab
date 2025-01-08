% Initialize variables to store traces for averaging
periopto_traces = cell(size(planefolders, 2), 1);  % For storing dFF traces
periopto_speed = cell(size(planefolders, 2), 1);   % For storing speed traces
pr_dir=uipickfiles; 
% Preallocate trace arrays
for allplanes = 1:size(planefolders, 2)
    periopto_traces{allplanes} = [];
    periopto_speed{allplanes} = [];
end

for days = days_check
    % Load suite2p data for current day
    dir_s2p = struct2cell(dir([pr_dir{days} '\**\suite2p']));
    planefolders = dir_s2p(:, ~cellfun(@isempty, regexp(dir_s2p(1, :), 'plane')));
    
    for allplanes = 1:size(planefolders, 2)
        % Load data for the current plane
        pr_dir2 = strcat(planefolders{2, allplanes}, '\plane', num2str(allplanes-1), '\reg_tif\');
        cd(pr_dir2);
        
        load('params');
        dFF = params.roibasemean3{1};
        dFF_base_mean = mean(dFF);
        dFF = dFF / dFF_base_mean;  % Normalize dFF
        
        % Get stimulation events
        abfstims = bwlabel(stims > 0.2);
        abfstims = abfstims > 0.2;
        abfrect = consecutive_stretch(find(abfstims));
        
        % Exclude very short stim periods
        abfrect(cellfun(@length, abfrect) < 50) = [];
        
        % DEBUG: Check if abfrect is empty
        if isempty(abfrect)
            disp(['No stimulation events found for plane ' num2str(allplanes) ' on day ' num2str(days)]);
            continue;
        end
        
        % Find peri-opto start periods
        xs = cellfun(@(x) utimedFF(x(1)), abfrect, 'UniformOutput', 1);
        
        % DEBUG: Check if xs is empty
        if isempty(xs)
            disp(['No opto start events for plane ' num2str(allplanes) ' on day ' num2str(days)]);
            continue;
        end
        
        for x = 1:length(xs)
            currx = find(timedFF >= (xs(x) - timewindow) & timedFF < (xs(x) + timewindow));
            currspeedx = find(utimedFF >= (xs(x) - timewindow) & utimedFF < (xs(x) + timewindow));
            
            % DEBUG: Check if currx or currspeedx are empty
            if isempty(currx)
                disp(['No valid dFF indices for plane ' num2str(allplanes) ' on day ' num2str(days)]);
            end
            if isempty(currspeedx)
                disp(['No valid speed indices for plane ' num2str(allplanes) ' on day ' num2str(days)]);
            end
            
            % Store dFF traces for each plane
            periopto_traces{allplanes} = [periopto_traces{allplanes}, dFF(currx)];
            
            % Store speed traces for each plane
            periopto_speed{allplanes} = [periopto_speed{allplanes}, forwardvelALL(currspeedx)];
        end
    end
end

% Now calculate the averages for each plane and plot
for allplanes = 1:size(planefolders, 2)
    % Check if traces were collected
    if ~isempty(periopto_traces{allplanes})
        avg_dFF = mean(periopto_traces{allplanes}, 2);  % Average across all trials
        avg_speed = mean(periopto_speed{allplanes}, 2);  % Average across all trials
        
        % Plot the average traces
        figure;
        subplot(2, 1, 1);
        yyaxis left;
        plot(timedFF(1:length(avg_dFF)) - timewindow, avg_dFF, '-', 'Color', planecolors{allplanes});
        ylabel('Average dFF');
        ylim([min(avg_dFF) max(avg_dFF)]);
        
        yyaxis right;
        plot(utimedFF(1:length(avg_speed)) - timewindow, avg_speed, 'k-');
        ylabel('Average Speed');
        ylim([min(avg_speed) max(avg_speed)]);
        yticks([0:25:75]);
        
        title(['Average PeriOpto Start Plane ' num2str(allplanes)]);
    else
        disp(['No data for plane ' num2str(allplanes)]);
    end
end
