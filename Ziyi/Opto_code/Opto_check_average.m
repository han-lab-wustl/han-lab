% Initialize variables to store traces for averaging
periopto_traces = cell(size(planefolders, 2), 1);  % For storing dFF traces
periopto_speed = cell(size(planefolders, 2), 1);   % For storing speed traces
pr_dir=uipickfiles; 

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
        dFF = dFF / dFF_base_mean;
        
        % Get stimulation events
        abfstims = bwlabel(stims > 0.5);
        abfstims = abfstims > 0.5;
        abfrect = consecutive_stretch(find(abfstims));
        
        % Exclude very short stim periods
        abfrect(cellfun(@length, abfrect) < 50) = [];
        
        % Find peri-opto start periods
        xs = cellfun(@(x) utimedFF(x(1)), abfrect, 'UniformOutput', 1);
        
        for x = 1:length(xs)
            currx = find(timedFF >= (xs(x) - timewindow) & timedFF < (xs(x) + timewindow));
            
            % Accumulate traces for dFF
            if isempty(periopto_traces{allplanes})
                periopto_traces{allplanes} = dFF(currx) - xs(x);
            else
                periopto_traces{allplanes} = [periopto_traces{allplanes}, dFF(currx) - xs(x)];
            end
            
            % Accumulate traces for speed
            currspeedx = find(utimedFF >= (xs(x) - timewindow) & utimedFF < (xs(x) + timewindow));
            if isempty(periopto_speed{allplanes})
                periopto_speed{allplanes} = forwardvelALL(currspeedx);
            else
                periopto_speed{allplanes} = [periopto_speed{allplanes}, forwardvelALL(currspeedx)];
            end
        end
    end
end

% Now calculate the averages for each plane and plot
for allplanes = 1:size(planefolders, 2)
    avg_dFF = mean(periopto_traces{allplanes}, 2);
    avg_speed = mean(periopto_speed{allplanes}, 2);
    
    % Plot the average traces
    figure;
    subplot(2, 1, 1);
    yyaxis left;
    plot(timedFF(1:length(avg_dFF)) - timewindow, avg_dFF, '-', 'Color', planecolors{allplanes});
    ylabel('Average dFF');
    ylim([0.85 1.2]);
    
    yyaxis right;
    plot(utimedFF(1:length(avg_speed)) - timewindow, avg_speed, 'k-');
    ylabel('Average Speed');
    ylim([-5 200]);
    yticks([0:25:75]);
    
    title(['Average PeriOpto Start Plane ' num2str(allplanes)]);
end
