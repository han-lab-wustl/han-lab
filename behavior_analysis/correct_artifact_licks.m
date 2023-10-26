function licks = correct_artifact_licks(ybinned,licks)
    % delete consecutive licks from signal
    x = 3; % here you can modify the cm

    % Take the difference (slope between points)
    diffL = diff(licks) == 1 ;
    
    % Pad zero out front
    diffL = [0 diffL];
    
    % keep only the starting point of the lick transients
    licks = licks.* logical(diffL);
    
    % delete all the licks before 'x' cm
    licks(ybinned<=x) = 0; 
end
