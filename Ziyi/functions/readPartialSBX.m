function data = readPartialSBX(filename, numFrames)
    % Efficient function to read specific rows and columns from multiple frames of an SBX file.
    % filename: string - the name of the SBX file.
    % numFrames: integer - the number of frames to read.
    % data: output - the read part of the data, returned as a 4D uint16 array.
    
    tic
    % Constants
    totalRows = 512;    % Total number of rows in the data
    totalCols = 796;    % Total number of columns in the data
    dataType = 'uint16'; % Data type used in the SBX file

    % Rows and columns of interest
    colsToRead = 741:796; % Read only columns from 741 to 796
    rowsToRead = 81:512;  % Read only rows from 81 to 512

    % Open the file
    fileId = fopen(filename, 'r');
    if fileId == -1
        error('File cannot be opened');
    end

    % Precompute byte offsets for efficiency
    bytesPerSample = 2; % Since data type is uint16
    bytesPerRow = totalCols * bytesPerSample;
    bytesPerFrame = totalRows * bytesPerRow;

    % Initialize the output data array
    data = zeros(length(rowsToRead), length(colsToRead), numFrames, 1, dataType);
    
    % Read data for each frame
    for frame = 1:numFrames
        % Calculate the byte offset to the start of the frame
        offset = (frame - 1) * bytesPerFrame;

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
    toc
end
