function [cleaneddff] = removeSquareWaveArtifacts(dff, samplingRate)
    cleaneddff = zeros(size(dff));
    for cll=1:size(dff,2)
    try
    originalSignal = dff(:, cll);
       % Parameters
    derivativeThreshold = std(diff(originalSignal))*3; % Define based on your data, try std(diff(originalSignal))*5 as a start
    minArtifactDuration = .02; % Minimum expected duration of an artifact in seconds
    minArtifactSamples = minArtifactDuration * samplingRate; % Convert duration to samples

    % Calculate signal derivative
    signalDerivative = diff(originalSignal);
    
    % Find where the derivative exceeds the threshold
    artifactIndices = find(abs(signalDerivative) > derivativeThreshold);
    
    % Group artifact indices into continuous segments
    % This step assumes artifacts are longer than 1 sample
    d = diff(artifactIndices);
    segmentBorders = find(d > 1);
    artifactStarts = [artifactIndices(1); artifactIndices(segmentBorders)+1];
    artifactEnds = [artifactIndices(segmentBorders); artifactIndices(end)];
    
    % Filter out short artifacts
    artifactDurations = artifactEnds - artifactStarts;
    validArtifacts = artifactDurations >= minArtifactSamples;
    artifactStarts = artifactStarts(validArtifacts);
    artifactEnds = artifactEnds(validArtifacts);
    
    % Initialize cleaned signal
    cleanedSignal = originalSignal;
    
    % Interpolate over artifacts
    for i = 1:length(artifactStarts)
        % Define interpolation range
        interpRange = artifactStarts(i):artifactEnds(i);
        
        % Linear interpolation
        if artifactStarts(i) > 1 && artifactEnds(i) < length(originalSignal)
            cleanedSignal(interpRange) = NaN; %interp1([artifactStarts(i)-1, artifactEnds(i)+1], ...
                                                  % [originalSignal(artifactStarts(i)-1), originalSignal(artifactEnds(i)+1)], ...
                                                  % interpRange, 'linear');
        end
    end
    cleaneddff(:,cll) = cleanedSignal;
    % figure; plot(cleanedSignal)
   
    catch
        cleaneddff(:,cll) = originalSignal;
    end
    end
end
