
% path = 'N:\Munni';
%%% MM DA
%%% population plots for early and late days for all the categories
%%%%%%Input: workspace folder 
%%%%%%output: 1) population mean 4 early and 4 late days actiivty of each
%%%%%%mouse individually
% % Input: workspaces folder
% % Output:Figures
% % single rew CS: First lick after reward
% %  	single rew CS
% % 	doubles: First lick after reward
% % stopping success trials: all stops
% % non rewarded stops
% %    rewarded stops
% % moving success trials: all motions
% % moving rewarded
% % moving unrewarded
% % unrewarded stops with licks
% % unrewarded stops without licks
% % 
% % Figure for combined planes (SO vs SP+SR+SLM) for GRABA and GRABDA-mutant
% % 
% % Figures for combined 
% % significance analysis:
% % comparison between GRABDA and GRABDA-mutant
% % 3 strategies:
% % All events combined early and late days
% % 4 early and late days of each mouse
% % Mean of 4early and late days each mouse
% % 
% % 
% % Section 2:
% % 
% % draw comparison between the categories. Let’s say single vs double CS…
% % for each mouse.




% close all
clear all
saving=0;
savepath='G:\dark_reward\dark_reward\figures ';
 filepath = 'F:\dark_reward_figures'
timeforpost = [0 1];
timeforpre=[-2 0];
timeforpost=[0 1.5];

    %% settings
    path='D:\workspaces_darkreward';%% WITH EARLIEST DAYS
    cd(path)
    workspace = {'156_dark_reward_AllDays_CutPlanes_workspace.mat','167_dark_reward_AllDays_CutPlanes_workspace.mat','168_dark_reward_workspace.mat','169_dark_reward_AllDays_CutPlanes_workspace.mat',...
        '171_dark_reward_AllDays_CutPlanes_workspace'};% '170_dark_reward_workspace.mat','179_dark_reward_workspace.mat',{'181_dark_reward_earlydays_workspace_00.mat','181_dark_reward_latedays_workspace_00.mat'}};
    
    figure;
    planecolors={[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
    
    for ws = 1:length(workspace)
        
        load(workspace{ws})
        means = cellfun(@(x) nanmean(reshape(x(40:42,:),[],1)),roi_dop_alldays_planes_periUS,'UniformOutput',1);
        sems = cellfun(@(x) nanstd(reshape(x(40:42,:),[],1))/sqrt(numel(x(40:42,:))),roi_dop_alldays_planes_periUS,'UniformOutput',1);
        subplot(2,3,ws)
        for p = 1:size(means,2)
            errorbar(means(:,p),sems(:,p),'Color',planecolors{p},'Capsize',0)
            hold on
        end
        
    end
        
