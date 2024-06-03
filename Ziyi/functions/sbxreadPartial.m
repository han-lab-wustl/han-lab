function x = sbxreadPartial(fname, k, N, rowRange, colRange)
    % img = sbxread(fname, k, N, rowRange, colRange)
    %
    % Reads from frame k to k+N-1 in file fname, only within the specified rows and columns.
    % Optimized to load all frames more efficiently.
    %
    % fname - the file name (e.g., 'xx0_000_001')
    % k     - the index of the first frame to be read. The first index is 0.
    % N     - the number of consecutive frames to read starting with k.
    % rowRange - the range of rows to read (e.g., 1:20)
    % colRange - the range of columns to read (e.g., 90:718)
    %
    % The function also creates a global 'info' variable with additional information about the file
    tic
    global info_loaded info

    % Load info if not already loaded or if it's a different file
    if(isempty(info_loaded) || ~strcmp(fname,info_loaded))
        sbxreadinfo(fname); 
    end

    if(isfield(info,'fid') && info.fid ~= -1)
        try
            % Determine number of bytes per sample and compute offsets
            bytesPerSample = 2; % uint16
            colSkip = info.sz(2) - colRange(end) + colRange(1) - 1;
            rowSkip = (info.sz(2) - colRange(end) + colRange(1) - 1) * (info.sz(1) - rowRange(end));

            % Allocate output array
            x = zeros([info.nchan, length(rowRange), length(colRange), N], 'uint16');

            % Compute the byte offset to start of the first frame to read
            fseek(info.fid, k * info.nsamples, 'bof');
            
            % Read data
            for i = 1:N
                for r = rowRange
                    % Skip to the beginning of the row of interest
                    fseek(info.fid, (r - 1) * info.sz(2) * info.nchan * bytesPerSample, 'cof');
                    
                    % Read the columns of interest
                    for c = 1:length(rowRange)
                        fseek(info.fid, (colRange(1) - 1) * info.nchan * bytesPerSample, 'cof');
                        x(:, c, :, i) = fread(info.fid, [info.nchan, length(colRange)], 'uint16=>uint16');
                        % Skip to the next row of interest
                        fseek(info.fid, colSkip * info.nchan * bytesPerSample, 'cof');
                    end
                    
                    % Skip to the start of the next frame of interest
                    if r < length(rowRange)
                        fseek(info.fid, rowSkip * bytesPerSample, 'cof');
                    end
                end
            end
            % Correct the data format and orientation
            x = intmax('uint16') - permute(x, [1 3 2 4]);
        catch
            error('Cannot read frame. Index range likely outside of bounds.');
        end
    else
        x = [];
    end
    toc
end


function sbxreadinfo(fname)
    global info
    load([fname '.mat']); % Load metadata
    info.fid = fopen([fname '.sbx'], 'rb');
end
