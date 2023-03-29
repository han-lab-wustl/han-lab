
% This function delete licks occurring before 'x' cm of the track.
% the imput required are the lick logical vector and the postion of the
% animal in the track.
%
% Created 01/27/21 by EB 


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
