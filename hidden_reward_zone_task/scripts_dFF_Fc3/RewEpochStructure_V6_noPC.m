%% Pyrs cells analysis
% Make sure the settings are correct.

% This script creates:
% ~rewEpoch~ variable, a cell structure where all the behavioral
% variables are separated by reward epochs learning phases {1,:}
% and probes {2,:}.
% ~all~ variable, where the spatial information for each cell is calculated
% across the whole recording session.
%
% User is required to be inside the folder with data analyzed after suite2p
% and aligned with behavior using VRselectstartendsplit.
%
% Functions required from the followiong path are:
% C:\Users\Han lab\Documents\MATLAB\Matlab_scripts\place_cell_scripts\image_analysis
% C:\Users\Han lab\Documents\MATLAB\HANLAB_SCRIPTS-master\Matlab_scripts\place_cell_scripts
% - Calc_Fc3_Reverse_Subtraction
% - get_spatial_tuning_all_cells
% - get_sorted_cells_idx
% - constrain_row

% You must find and download : https://www.mathworks.com/matlabcentral/answers/254048-how-can-i-delete-variables-in-my-mat-file-without-loading-them-into-matlab-7-2-r2006a#:~:text=MATLAB%20itself%20does%20not%20offer,variable%20from%20a%20MAT%20file.
% - RemoveVariableFromMatFile
% Note: remember to change the path on line 85 of this function

% You'll find at the end of this script
% - get_reward_location_Idxs
% - correct_artifact_licks
%
%
% 01/15/22 EB
% eleonora.bano@wustl.edu
% *************************************************************************
% clear the environment and load your neural and behavioral data aligned
if exist('done','var') == 0 || done == 1
    clc
    clear
end

% Settings-----------------------------------------------------------------
done = 0;
days = 1; %E144 end d39 days you are recrding from (Fall_D11,Fall_D4,Fall_D5) will be listed as [11 4 5]
settings.Fs = 32; % reconrding frequency
settings.binSize = 1; % cm
settings.track_length = 180; % cm
settings.numprobe = 3; % same from VR.settings, how many probe trials in these recordings
settings.dir{1} = 'J:\Imaging\221107_WB\suite2p\plane0';
settings.dir{2} = 'J:\Imaging\221107_WB\suite2p\plane0';

if exist('Fc3','var') == 1 && done == 0
    settings.Fc3 = Fc3;
    settings.dFF = dFF;
end
%--------------------------------------------------------------------------

for day = 1 : numel(days)
    % clean the anvironment and load your variables
    
    clc
    close all
    clearvars -except days day settings done
    if numel(num2str(days(day)))>1
        in = 2;
    else
        in = 1;
    end
    idx = strfind(settings.dir{in},'X');
    dir = settings.dir{in};
    dir(idx)=string(days(day));
    % check the name of yor Fall, if multiple days are concatenated, name is in
    % format Fall_daynum
    % filename = ['Fall_D' num2str(days(day))];
    filename = 'Fall';
    directory = dir;
    cd(dir)
    listing = string(ls(directory));
    startIndex = regexp(listing,filename);
    if numel(days) == 1
        if sum(cell2mat(startIndex))>0
            load(filename)
        else
            filename = 'Fall';
            load(filename)
        end
    else
        load(filename)
    end
    
    if exist('rewEpochV6_noPC','var')
        prompt = 'This file has been analyzed already. Do u want to reanalyze it? 0-no ; 1-yes = ';
        keepGoing = input(prompt);
        if keepGoing == 0
            return
        elseif keepGoing == 1
            mex('C:\Users\Han lab\Documents\MATLAB\EB scripts\HRZ basic functions\RemoveVariableFromMatFile.c') % change path here if necessary
            RemoveVariableFromMatFile([filename '.mat'],'rewEpochV6_noPC','all')
        end
    end
    clearvars startIndex listing directory
    
    who = whos;
    % get the variables we are interested in
    names = {who.name}';
    names = [names; 'timedFFminutes'; 'dFF'; 'Fc3']; %#ok<AGROW>
    [~,legitVarsIdx] = find(contains(string({who.class}),["logical","single","double"]));
    % note: using the 'class' information should account for insertions of new variables from VRselectstartendsplit
    [~,idx2delete] = find(contains(string({who.name}),["trialnum","day","days"]));
    % trialnum variable to be the first one to be processed.
    [~,idxTrials] = find(contains(string({who.name}),"trialnum"));
    % combine the indeces I'm interested in and move the index of the trialNum
    % variable at the beginning
    legitVarsIdx(legitVarsIdx==idxTrials) = [];
    legitVarsIdx = [idxTrials legitVarsIdx numel(names)-2 numel(names)-1 numel(names)]; %#ok<AGROW>
    
    timedFFminutes = timedFF / 60; % convert from seconds to minutes
    
    if exist('iscell','var')
        % work only with the cells visually selected in suite2p
        cells = find(logical(iscell(:,1)));
    else
        cells = 1:size(F,1);
    end
    
    % call the settings
    Fs = settings.Fs ;
    binSize = settings.binSize ; % cm
    track_length = settings.track_length ; % cm
    numprobe = settings.numprobe; % same from VR.settings
    nBins = track_length/binSize ;
    
    
    % creates dFF and Fc3 variables
    % inputs
    position = ybinned ;
    
    % outputs
    if ~isfield(settings,'Fc3')
        [dFF,Fc3] = Calc_Fc3_Reverse_Subtraction(F(cells,:),Fneu(cells,:),Fs);
        dFF = dFF';
        Fc3 = Fc3';
    else
        dFF = settings.dFF;
        Fc3 = settings.Fc3;
    end
    
    % save neuronal data for the whole session
    all.dff = dFF;
    all.Fc3 = Fc3;
    all.Spks = spks(iscell(:,1)==1,:);
    
    %% calculates place cells info realtively to the whole session
    % all structure containing info across the whole session is created
    
    all.cell_activity = ...
        get_spatial_tuning_all_cells(Fc3',position,Fs,nBins,track_length);
    [all.cellIdx, all.maxBin] = ...
        get_sorted_cells_idx(all.cell_activity);
    
    % plot the tuning curve of all the selected cells
    figAll = figure();
    imagesc(1:10)
    imagesc(constrain_row(all.cell_activity(all.cellIdx,:)));
    title ('tuning curve of all the cells across whole recording')
    
    %% make rewEpoch structure
    
    % find when the reward location change and related infos
    [rewLocIdx, rewLoc, rLn] = get_reward_location_Idxs(changeRewLoc);
    
    % get the licks corrected
    licks = correct_artifact_licks(ybinned,licks);
    
    % initialize your structure
    rewEpoch = cell(2,rLn);
    
    for rL = 1 : rLn
        % for each reward epoch...
        if  rL == 1
            incipit = 'Hello (: ';
        else
            incipit = 'now ';
        end
        
        % define the indeces of the current epoch
        currentEpochIdxs = rewLocIdx(rL):rewLocIdx(rL+1)-1;
        a = 0; %sometimes, the first datapoint of rew epoch belongs to the previous one, probably due to adjustments in VRexperiment code
        while trialnum(currentEpochIdxs(1))>numprobe+1
            a = a+1;
            currentEpochIdxs = []; %#ok<NASGU>
            currentEpochIdxs = (rewLocIdx(rL)+a):rewLocIdx(rL+1)-1;
        end
        
        
        % identify each trial
        trialnum_singleEpoch = trialnum(currentEpochIdxs);
        trials = trialnum_singleEpoch+1; % so we never have zeros as indeces
        trials(diff(trialnum_singleEpoch)==1) = 0;
        split = bwlabel(trials);
        probIdx = (split(trialnum_singleEpoch<=numprobe-1));
        
        % find the index that separate probes from trials
        split = find(trialnum(currentEpochIdxs) > numprobe-1,1, 'first');
        probeLocIdx = (rewLocIdx(rL)-1)+split;
        if isempty(probeLocIdx) && ~isempty(currentEpochIdxs)
            % if there are only probe trials
            probeLocIdx = rewLocIdx(end);
            onlyProbe = 1; % dummy saying there are only probes in this epoch
        else
            onlyProbe = 0;
        end
        
        % only the learning phase of the current epoch
        learningPhaseIdx = probeLocIdx:currentEpochIdxs(end);
        
        % only the probe phase of the current epoch
        probePhaseIdx = currentEpochIdxs(1):probeLocIdx-1;
        
        if ~isempty(probeLocIdx)
            % reward location one may or may not have probe trials at the
            % beginning
            if onlyProbe == 1
                subEpochs = 2;
                n = 1; % update waiting bar color filling
                numpr = sum(diff(trialnum(probePhaseIdx)))+1;% how many probe trials
            else
                subEpochs = 1:2;
                n = 0.5; % update waiting bar color filling
                numpr = sum(diff(trialnum(probePhaseIdx)))+1;% how many probe trials
            end
        else
            subEpochs = 1;
            n = 1; % update waiting bar factor
        end
        
        for kk = subEpochs
            % first row correspond to learning epochs, second to probe epochs
            if kk == 1
                % settings for learning epochs
                index = learningPhaseIdx;
                c = numprobe-1; % correction to fill up rewEp structure
                trNumI = 1:max(trialnum(index))-c;
                % create the waitbar message
                tit = ' Learning subEpoch!';
                message = [incipit 'working on RewardEpoch' num2str(rL) ':' tit];
            elseif kk == 2
                % settings for probe epochs
                index = probePhaseIdx;
                c = numprobe-numpr-1;
                trNumI = 1:numpr;
                % create the waitbar message
                tit = ' Probe subEpoch!';
                message = [incipit 'working on RewardEpoch' num2str(rL) ':' tit];
            end
            
            % create the waitbar
            if exist('bar','var')
                waitbar(rL/rLn,bar,message);
            else
                bar = waitbar(rL/rLn,message);
            end
            
            for varnameIdx = legitVarsIdx
                % separate all the viarables by epochs
                varname = names{varnameIdx};
                currVar = evalin('base',varname);
                
                if size(currVar,2)> numel(currentEpochIdxs)
                    % process for timeseries variables
                    rewEpoch{kk,rL}.(varname) = currVar(:,index);
                    
                    %% Create the binnedTrial structure
                    for jj = trNumI
                        % get the datapoint belonging to each trial
                        currTrIdx = find(rewEpoch{kk,rL}.trialnum==(jj+c));
                        if numel(currTrIdx)>4 % eliminate one timepoint trials artefects
                            % teleport datapoint correction
                            currTrIdx = currTrIdx(2:end);
                            % create the binnedTrial structure
                            rewEpoch{kk,rL}.binnedTrial.(varname){jj} =...
                                rewEpoch{kk,rL}.(varname)(:,currTrIdx);
                            if varnameIdx==legitVarsIdx(end) && numel(rewEpoch{kk, rL}.binnedTrial.trialnum(:))>1
                                % trial by trial metrics
                                fc3 = rewEpoch{kk,rL}.binnedTrial.Fc3{jj} ;
                                position = rewEpoch{kk,rL}.binnedTrial.ybinned{jj};
                                
                                rewEpoch{kk,rL}.binnedTrial.cell_activity{jj} = ...
                                    get_spatial_tuning_all_cells(fc3',position,Fs,nBins,track_length);
                                [rewEpoch{kk,rL}.binnedTrial.cellIdx{jj}, rewEpoch{kk,rL}.binnedTrial.maxBin{jj}] = ...
                                    get_sorted_cells_idx(rewEpoch{kk,rL}.binnedTrial.cell_activity{jj});
                                rewEpoch{kk,rL}.cell_info{jj} =...
                                    get_spatial_info_all_cells(fc3',position,Fs,nBins,track_length);
                                rewEpoch{kk,rL}.cell_sparsity{jj} = ...
                                    get_spatial_sparsity_all_cells(fc3',position,Fs,nBins,track_length);
                                rewEpoch{kk,rL}.com {jj}= calc_COM(rewEpoch{kk,rL}.binnedTrial.cell_activity{jj});
                            end
                            
                        end
                    end
                end
            end
        end
    end
    
    % convert to table
    rewEpochV6_noPC = cell2table(rewEpoch,...
        'RowNames',{'Learning Epochs' 'Probe Epochs (@epoch start)'});
    
    % save data in the current folder
    save([dir '\Fall'],'all','rewEpochV6_noPC','-v7.3','-append')
    save([dir '\rewEpochV6_noPC'],'rewEpochV6_noPC','-v7.3','-nocompression')
    done = 1;
    delete(bar)
end



function [rewLocInds, rewLocs, rewLocN] = get_reward_location_Idxs(changeRewLoc)
% get the indexes of when the reward location changes
rewLocInds = [find(changeRewLoc>0) length(changeRewLoc)];

% values of the reward locations
rewLocs = changeRewLoc(rewLocInds);

% number of reward locations
rewLocN = size(rewLocInds,2)-1;
end



