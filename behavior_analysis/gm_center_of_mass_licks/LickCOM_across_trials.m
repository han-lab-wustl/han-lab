%% COM general view
% zahra's mods

% generates three plots
% COM plot with some modifications
% ratio of peri-rewad licks to non-peri
% lick distance from rew start done by lick, not trial.
% for ratio and lick dist, removes no lick trials for t-test
% added bunch of variables to control number of trials and peri-rew window
%add quick lick correction with diff to eliminate consecutive "licks"
% Now lick distance from start of rew zone (=0)


%% ------------------------------------------------------------------------
clear all %clean up the environment
close all;
mouse_name = "z8";
fltype = 'daypth'; %vs. day path
if fltype == 'select'
    [filename,filepath] = uigetfile('*.mat','MultiSelect','on');
    days = filename;
else
    % ZD added for loop for multiple days
    %     days = [55:75];
    %     days = [4:7,9:11];
    %     days = [62:67,69:70,72:74,76,81:85];
    days = [32];
    %     src = "Z:\sstcre_imaging";
    %     src = "X:\pyramidal_cell_data";
    %     src = "Y:\sstcre_imaging";
    src = 'X:\vipcre';
end
for dy=1:length(days)
    if fltype == 'select'
        clearvars -except mouse_name filepath filename dy days fltype pptx
        file = fullfile(filepath,filename{dy});
    else
        clearvars -except mouse_name days src dy fltype pptx % Zahra added for for loop
        daypth = dir(fullfile(src, mouse_name, string(days(dy)), "behavior", "vr\*.mat"));
        %         daypth = dir(fullfile(src, mouse_name, string(days(dy)), "*.mat"));
        filename{dy} = daypth.name;
        filepath{dy} = daypth.folder;
        file=fullfile(filepath{dy},filename{dy});
    end
    load(file) %load it
    if length(find(VR.changeRewLoc)) > 1
        try % based on 2 diff ways to read files
            cd (filepath{dy}); %EH set path
        catch
            cd (filepath);
        end
        COMgeneralanalysis
        set(gca,'FontName','Arial')  % Set it to arail
        % ZD - save to ppt
        fig = figure('Renderer', 'painters', 'Position',[10 10 300 500]);
        %COM plots
        for tt = 1:length(RewLoc) % for the three reward locations..
            if ~isempty(allCOM{tt})
                subplot(length(RewLoc),1,tt) % blue normalized lick scatterplot all the trials, succesfull or not are considered
                % subplot(2,1,tt)
                hold on
                %             line([1,length(allCOM{tt})],[-7.5,-7.5],'LineStyle','--','LineWidth',1.5,'Color',	[0 0 0 0.4]) % dashed line is plotted at the nomalized COM (y=0 always) - 7.5
                line([1,length(allCOM{tt})],[0,0],'LineStyle','--','LineWidth',1.5,'Color',	[0 0 0 0.4]) %EH line at 0 for start of rew zone
                for i=1:numel(trialyposlicks{tt}) % the x axis indicates the trial number
                    % (eg. if trial 4 has no licks, the location 4 on the x axes will appear empty and the next trial (trial 5)
                    % will be plotted at x positon 5)
                    scatter(ones(size(trialyposlicks{tt}{i})).*i,(trialyposlicks{tt}{i})-RewLocStart(tt),38,[0 0.4470 0.7410],'filled')
                    xlabel('Trial')
                    ylabel('\Delta Rew. Loc. (cm)')
                end
                alpha(.05)
                scatter((1:length(allCOM{tt})),allCOM{tt},50,'g','filled')
                scatter(find(failure{tt}),allCOM{tt}(find(failure{tt})),50,'k','filled')
                title([ 'Rew. Loc. ' num2str(tt) ' = ' num2str(RewLocStart(tt))])
                xlim([1, length(allCOM{tt})])
                xticks([1:2:length(allCOM{tt})])
                xticklabels([1:2:length(allCOM{tt})])
            end
        end        
    end
    sgtitle(days(dy))
end