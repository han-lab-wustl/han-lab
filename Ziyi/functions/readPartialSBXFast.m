function data = readPartialSBXFast(filename, startFrame, endFrame)
    % Efficient function to read specific rows and columns from a range of frames of an SBX file.
    % filename: string - the name of the SBX file.
    % startFrame: integer - the starting frame (inclusive).
    % endFrame: integer - the ending frame (inclusive).
    % data: output - the read part of the data, returned as a 4D uint16 array.

    % Constants
    totalRows = 512;    % Total number of rows in the data
    totalCols = 796;    % Total number of columns in the data
    dataType = 'uint16'; % Data type used in the SBX file

    % Rows and columns of interest
    % colsToRead = 90:718; % Read only columns from 90 to 718
    % rowsToRead = 1:20;   % Read only rows from 1 to 20
    colsToRead = 740:796; % Read only columns from 90 to 718
    rowsToRead = 81:512;   % Read only rows from 1 to 20

    % Validate frame range
    if startFrame > endFrame
        error('Start frame must be less than or equal to end frame');
    end

    % Open the file
    fileId = fopen(filename, 'r');
    if fileId == -1
        error('File cannot be opened');
    end

    % Number of frames to read
    numFrames = endFrame - startFrame + 1;

    % Precompute byte offsets for efficiency
    bytesPerSample = 2; % Since data type is uint16
    bytesPerRow = totalCols * bytesPerSample;
    bytesPerFrame = totalRows * bytesPerRow;

    % Initialize the output data array
    data = zeros(length(rowsToRead), length(colsToRead), numFrames, 1, dataType);
    
    % Read data for each frame within the range
    for frame = 1:numFrames
        actualFrame = startFrame + frame - 1;  % Translate to actual frame index
        % Calculate the byte offset to the start of the actual frame
        offset = (actualFrame - 1) * bytesPerFrame;

        % Read the rows of interest
        for i = 1:length(rowsToRead)
            row = rowsToRead(i);
            % Calculate the offset for the current row
            currentRowOffset = offset + (row - 1) * bytesPerRow + (colsToRead(1) - 1) * bytesPerSample;
            
            % Position the file read pointer
            fseek(fileId, currentRowOffset, 'bof');
            
            % Read the relevant columns from the current row
            data(i, :, frame, 1) = fread(fileId, length(colsToRead), dataType);
        end
    end

    % Close the file
    fclose(fileId);
end
