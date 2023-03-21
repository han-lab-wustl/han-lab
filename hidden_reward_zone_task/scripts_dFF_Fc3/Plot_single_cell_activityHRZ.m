%% Plot single cell activity

% This script plot the cells from the rewardEpoch structure created with
% the script rewardEpochStructure that is found in:
% C:\Users\Han lab\Documents\MATLAB\EB scripts

% The first row, shows the raw activity across the whole recording, each
% trial is plotted individually. Solid line at the bottom, represnt the
% learning subEpoch, dashed line represent probe subEpoch.
% The function 
% - distinguishable_colors 
% is required.
%
% 0/28/21 EB 
% eleonora.bano@wustl.edu
% *************************************************************************
% clear the environment and load your neural and behavioral data aligned

clc
clear

condition = 'fam';
saving = 0 ; % option to save the figures
sFormat = '.png'; % saving format 
                  % PS: check 'saveas' docs for option availables

load('structures_analyzed_data.mat')
nEpochs = size(rewEpoch,2);

cell_num =1%:149;% rewEpoch{1, 1}.place_cells{1, 1};
referencelinePosition = -3; %position line summarizing reward epochs

% each reward epoch will have its color
color = get_color_HRZ(condition);
colors = [];
for clrs = 1:nEpochs
    colors(clrs,:) = color{clrs};
end
% 
% 
% 
% for c = 1:nEpochs
%     rndIdx = randperm(3,2);
%     a = [round(rand,2) round(rand,2)];
%     colors(c,rndIdx) = a;
% end

pp = zeros(nEpochs,1); % variable used for indexing subplots

for i=cell_num
    
    figure('units','normalized','outerposition',[0 0 0.5 1])
    % define size and opening location of the figure
    
    
    for rL = 1:nEpochs
        
        % determine if we have probe trials in the current reward epoch
        if ~isempty(rewEpoch{2,rL})
            rowIdx = [2 1];
        else
            rowIdx = 1;
        end
        
        for stI = rowIdx
            % get the data for each subEpoch of rewEpoch
            numTrials = length(rewEpoch{stI,rL}.binnedTrial.dFF);
            
            % for both, probe trials and learning trials
            lineStart = min(rewEpoch{stI,rL}.timedFFminutes); %line under beh
            lineEnd = max(rewEpoch{stI,rL}.timedFFminutes);
            subplot(nEpochs+1,3,[1 2 3])
            hold on
            if stI == 2
                % different setting for probe of learning
                linex = (lineStart:0.2:lineEnd);
                liney = ones(size(linex));
                plot(linex,liney*referencelinePosition,'.', 'LineWidth',4, 'Color',colors(rL,:))
            else
                plot([lineStart lineEnd],[referencelinePosition referencelinePosition], 'LineWidth',4, 'Color',colors(rL,:))
            end
            
            for jj = 1:numTrials
                % get the data for each trial
                
                lidx = rewEpoch{stI,rL}.binnedTrial.licks{1, jj}==1;
                ridx = rewEpoch{stI,rL}.binnedTrial.rewards{1, jj}==1;
                lengthOfRecording = max(rewEpoch{1,nEpochs}.timedFFminutes);
                f = 60; % to make bigger the neural signal
                
                % variables that we are plotting
                time = rewEpoch{stI,rL}.binnedTrial.timedFFminutes{1, jj};
                ypos = rewEpoch{stI,rL}.binnedTrial.ybinned{1, jj};
                dff = rewEpoch{stI,rL}.binnedTrial.Fc3{1, jj}(i,:);
                dffYpos = ypos(dff>0);
                timeDffYpos = time(dff>0);
                licksypos = ypos(lidx);
                rewardsypos = ypos(ridx);
                timelicks = time(lidx);
                timerewards = time(ridx);
                
                subplot(nEpochs+1,3,[1 2 3])
                plot(time, ypos,'color', [0.7 0.7 0.7])
                title(['Cell ' num2str(i)],'FontSize',18,'FontName','Arial')
                hold on
                scatter(timeDffYpos,dffYpos,3.5,[0.6 0 0.6],'filled')
                scatter(timelicks,licksypos,'.r')
                scatter(timerewards,rewardsypos,'.b')
                plot(time,dff*f,'k')

                
                ylim([-5 187])
                xlim([1 lengthOfRecording])
                xticks([1+.3 lengthOfRecording-.3])
                xticklabels({1, round(lengthOfRecording)})
                yticks([0 180])
                set(gca, 'TickLength',[0 0])
                box off
                clearvars  time ypos dff licksypos rewardsypos timelicks timerewards
                ylabel(' Position along the track - cm','FontSize',10,'FontName','Arial')
                xlabel('Time - minutes','FontSize',10,'FontName','Arial')
            end
            if stI ==1
                % subplots for learning epochs only
                if rL == 1
                    pp(rL) = 4; % previous position
                else
                    pp(rL) = pp(rL-1)+3; %updated subplot position
                end
                subplot(nEpochs+1,3,pp(rL))
                numofTr = length(rewEpoch{1,rL}.binnedTrial.dFF);
                col = colors(rL,:);
                [~,rd] = min(col);
                for jj = 1:numofTr
                    % plot dFF for each trial
                    col(rd) = 1*jj/numofTr;
                    %create shaded colors
                    plot (rewEpoch{1,rL}.binnedTrial.ybinned{1,jj},rewEpoch{1,rL}.binnedTrial.dFF{1,jj}(i,:)+jj,'Color',col)
                    hold on
                end
                ylabel('dFF / Trial','FontSize',10,'FontName','Arial')
                xlabel('Track Length - cm','FontSize',10,'FontName','Arial')
                yticks([1 length(rewEpoch{1,rL}.binnedTrial.ybinned)])
                yticklabels({'',num2str(length(rewEpoch{1,rL}.binnedTrial.ybinned))})
                a = get(gca,'YTickLabel');
                set(gca,'YTickLabel',a,'fontsize',10)
                set(gca, 'TickLength',[0 0])
                xticks([0 180])
                box off
            end
            
        end
    end
    
    
    idxMat = zeros(numel(pp),2);
    %index average activity subplot
    for im = 1:numel(pp)
        idxMat(im,1:2) = [pp(im)+1 pp(im)+2];
    end
    idxMat = reshape(idxMat,1,numel(idxMat));
    
    
    subplot(nEpochs+1,3,idxMat)
    % add the average activity subplot
    for in = 1:nEpochs
        ylabel('dFF','FontSize',10,'FontName','Arial')
        xlabel('Track Length - cm','FontSize',10,'FontName','Arial')
        plot(rewEpoch{1,in}.cell_activity(i,3:177),'Color',colors(in,:),'LineWidth',3)
        
        hold on
        xticks(0:30:180)
        set(gca, 'TickLength',[0 0])
        box off
        
    end
    if saving == 1
         saveas(gcf,['cell' num2str(i) sFormat])
    end
        
end
