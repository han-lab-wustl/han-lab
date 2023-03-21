%% final figures:
% Saves in Ed's figure folder.
%% ========================================================================
% Settings:
epochstructure = 'cs_table_last8trials.mat';
cd ('Z:\HRZ_master_output8trials')
load(epochstructure)
load('Z:\HRZ_master_outputCS_table_all_trials\CS_trial_by_trial_probes_all.mat')
% Summary=readtable("J:\optodata\E186\Summary.xlsx");

splitted_epochstructure_name = split(epochstructure,'_');

UL_track = 180;
bin_size = 3.3; %cm remember to adjust for gain
%==========================================================================

for this_comparison = 1: height(cs_table)
    
    mouse_cd = cs_table.mouse(this_comparison);
    day_cd = cs_table.Day(this_comparison);
    experiment_type = 'tracking';

    EP1 = cs_table.comparison_type (this_comparison, 1);
    EP2 = cs_table.comparison_type (this_comparison, 2);

    % histogram plot
    figure('Renderer', 'painters', 'Position', [20 20 1000 700])
    fig = tiledlayout('flow');
    nexttile
    hold on

    shuffled_distribution = cell2mat(cellfun(@(x) cell2mat(x),cs_table.shuffled_CS,UniformOutput=false));
    real_distribution = cell2mat(cs_table.real_CS);

    h1 = histogram(shuffled_distribution,'Normalization','probability','BinWidth',0.025);
    h1.FaceColor = [0 0 0];
    h1.EdgeColor = [1 1 1];
    xline(median(shuffled_distribution,'all'),'-k')

    h2= histogram(real_distribution,'Normalization','probability','BinWidth',0.025);
    h2.FaceColor = [1 0 0];
    h2.EdgeColor = [1 1 1];
    xline(median(real_distribution,'all'),'-r')

    axis padded
    %                             axis off

    title(fig,[char(mouse_cd) '; ' char(day_cd) ' ' experiment_type],[ 'ep' num2str(EP1) ' vs. ep' num2str(EP2) ' RankSum p = ' num2str(cs_table.RankSumP(this_comparison)) ' ; ' splitted_epochstructure_name{3} ' trials comparison'])
    %                             legend({'real',['median real = ' num2str(median(real_distribution,'all'))],...
    %                                 'shuffled',['median shuffle = ' num2str(median(shuffled_distribution,'all'))]},'Location','northwest')
    legend({'shuffled',['median shuffle = ' num2str(median(shuffled_distribution,'all'))],...
        'real',['median real = ' num2str(median(real_distribution,'all'))]},'Location','northwest')       

    % ----

    % imagesc plot
    this_mouse = find(cellfun(@(x) strcmp(x,mouse_cd),table_trial_by_trial_probes_all_CS.mouse));
    this_day = find(cellfun(@(x) strcmp(x,day_cd),table_trial_by_trial_probes_all_CS.trial_day));
    these_epochs = find(table_trial_by_trial_probes_all_CS.trial_epoch==EP1 | table_trial_by_trial_probes_all_CS.trial_epoch==EP2); 
    this_EP1 =  find(table_trial_by_trial_probes_all_CS.trial_epoch==EP1);
    this_EP2 =  find(table_trial_by_trial_probes_all_CS.trial_epoch==EP2);
    this_mouse_this_day = intersect(this_mouse,this_day);
    this_mouse_this_day_these_epochs = intersect(this_mouse_this_day,these_epochs);
    this_mouse_this_day_these_epochs = this_mouse_this_day_these_epochs';
    this_mouse_this_day_EP1 = intersect(this_mouse_this_day,this_EP1);
    this_mouse_this_day_EP2 = intersect(this_mouse_this_day,this_EP2);

    rew_location1 = nanmean(table_trial_by_trial_probes_all_CS.rew_location(this_mouse_this_day_EP1),"omitnan");
    rew_location2 = mean(table_trial_by_trial_probes_all_CS.rew_location(this_mouse_this_day_EP2),"omitnan");
    rew_location1_index = rew_location1\bin_size;
    rew_location2_index = rew_location2\bin_size;

   
    [~,max_bin1] = max(cs_table.tuning_curves_for_this_comparison{this_comparison,1},[],2);
   
    [~,max_bin2] = max(cs_table.tuning_curves_for_this_comparison{this_comparison,2},[],2);
    [~,sorted_idx] = sort(max_bin1);
    [~,sorted_idx2] = sort(max_bin2);


    TC_imagesc = [cs_table.tuning_curves_for_this_comparison{this_comparison,1}(sorted_idx,:) ...
        cs_table.tuning_curves_for_this_comparison{this_comparison,2}(sorted_idx,:)];

    this = nexttile;

    imagesc(normalize(TC_imagesc ,2,'range'))
    colormap(this,turbo)
    xlabel('cm')
    ticks = [rew_location1_index rew_location2_index+(UL_track/bin_size) ];
    ticks = sort(ticks);
    ticks_labels = [rew_location1 rew_location2];
    xticks(ticks)
    xticklabels(ticks_labels)
    xline(rew_location1_index ,'--w','LineWidth',1.5)
    xline(rew_location2_index + (UL_track/bin_size),'--w','LineWidth',1.5)
    title(this,{splitted_epochstructure_name{3};['epoch ' num2str(EP1) ';epoch ' num2str(EP2)]},'Interpreter','none')

    axis square
    % ----

    % CS plot
    

    CS_mat = nan(numel(this_mouse_this_day_these_epochs));
    ind1 = 0;
    for this_trial1 = this_mouse_this_day_these_epochs
        ind1 = ind1 +1;
        pop_vector1 = table_trial_by_trial_probes_all_CS.tuning_curves{this_trial1};
        pop_vector1 = reshape(pop_vector1 ,1,numel(pop_vector1 ));

            ind2 = 0;
        for this_trial2 = this_mouse_this_day_these_epochs
            ind2 = ind2 +1;
            pop_vector2 = table_trial_by_trial_probes_all_CS.tuning_curves{this_trial2};
            pop_vector2 = reshape(pop_vector2 ,1,numel(pop_vector2 ));

            CS_mat(ind1,ind2) = getCosineSimilarity(pop_vector1,pop_vector2);
        end
    end

    Y = CS_mat.';
    CS = reshape(Y(eye(size(Y))'~=1),size(Y,1)-1,size(Y,1)).';
    
    this = nexttile;
    imagesc(CS)
    x = table_trial_by_trial_probes_all_CS.trial_comparison(this_mouse_this_day_these_epochs,1);
    y = table_trial_by_trial_probes_all_CS.trial_epoch(this_mouse_this_day_these_epochs,1);
    xticks(1:5:ind1-1)
    xticklabels(x(1:5:ind1-1))
    yticks(1:5:ind2)
    yticklabels(y(1:5:ind2))
    xline(find(x==0)-0.5,'--r','LineWidth',1.5)
    xline(find(x==3)-0.5,'--r','LineWidth',1.5)
    yline(find(x==0)-0.5,'--r','LineWidth',1.5)
    yline(find(x==3)-0.5,'--r','LineWidth',1.5)
    legend({'probe trials'},'Location','northeastoutside')
    colormap(this,"bone")
    colorbar
    title(this,{'trial by trial CS'},{['epoch ' num2str(EP1) ';epoch ' num2str(EP2)],'diag. removed'},'Interpreter','none')
    xlabel('trialnum')
    ylabel('epoch')
    
    axis square
    
    % imagesc2

    TC_imagesc = [cs_table.tuning_curves_for_this_comparison{this_comparison,1}(sorted_idx2,:) ...
        cs_table.tuning_curves_for_this_comparison{this_comparison,2}(sorted_idx2,:)];

    this = nexttile;

    imagesc(normalize(TC_imagesc ,2,'range'))
    colormap(this,turbo)
    xlabel('cm')
    ticks = [rew_location1_index rew_location2_index+(UL_track/bin_size) ];
    ticks = sort(ticks);
    ticks_labels = [rew_location1 rew_location2];
    xticks(ticks)
    xticklabels(ticks_labels)
    xline(rew_location1_index ,'--w','LineWidth',1.5)
    xline(rew_location2_index + (UL_track/bin_size),'--w','LineWidth',1.5)
    title(this,{splitted_epochstructure_name{3};['epoch ' num2str(EP1) ';epoch ' num2str(EP2)]},'Interpreter','none')

    axis square

    % imagesc activity
    figure('Renderer', 'painters', 'Position', [800 100 1000 500])
    fig2 = tiledlayout(2,1);
    EPs_table = table_trial_by_trial_probes_all_CS(this_mouse_this_day_these_epochs,:);
    datapoint_per_trial = cell2mat(cellfun(@(x) numel(x),EPs_table.ypos,'UniformOutput',false));
    probes1 = find((EPs_table.trial_comparison(:,1)==0));
    probes2 = find((EPs_table.trial_comparison(:,1)==2));
    probe_start = [];
    for this_probe = 1:numel(probes1)
        if probes1(this_probe)~=1
         probe_start(this_probe) = sum(datapoint_per_trial(1:(probes1(this_probe))-1));
        end
    end
    probe_end = [];
     for this_probe = 1:numel(probes2)
        if probes2(this_probe)~=1
         probe_end(this_probe) = sum(datapoint_per_trial(1:(probes2(this_probe))));
        end
    end
    activity = EPs_table.dff'; activity = cell2mat(activity);
    activity1 = normalize(activity(sorted_idx,:),2,'range');
    activity2 = normalize(activity(sorted_idx2,:),2,'range');
    this = nexttile;
    imagesc(activity1)
    xline(probe_start ,'--w','LineWidth',1.5)
    xline(probe_end ,'--w','LineWidth',1.5)
    x = xticks;
    nexttile
    imagesc(activity2)
    xline(probe_start ,'--w','LineWidth',1.5)
    xline(probe_end ,'--w','LineWidth',1.5)
    x_min = x(end)/(Settings.Fs*60);
    xlabel([ 'duration = ' num2str(x_min) ' min'])
    colormap(turbo)
    title(this,{['ordered max ' splitted_epochstructure_name{3}];['epoch ' num2str(EP1) ';epoch ' num2str(EP2)]},'Interpreter','none')

    % ----
    try
    saveas(fig,['J:\optodata\E186\221212\matlab_output\figures\mat\' char(mouse_cd) char(day_cd) '_ep' num2str(EP1) '_vs_ep' num2str(EP2)],'fig')
    saveas(fig,['J:\optodata\E186\221212\matlab_output\figures\png' char(mouse_cd) char(day_cd) '_ep' num2str(EP1) '_vs_ep' num2str(EP2) '.png'],'png')
    saveas(fig,['J:\optodata\E186\221212\matlab_output\figures\svg\' char(mouse_cd) char(day_cd) '_ep' num2str(EP1) '_vs_ep' num2str(EP2) '.svg'],'svg')
    saveas(fig2,['J:\optodata\E186\221212\matlab_output\figures\mat\' char(mouse_cd) char(day_cd) '_ep' num2str(EP1) '_vs_ep' num2str(EP2) 'activity'],'fig')
    saveas(fig2,['J:\optodata\E186\221212\matlab_output\figures\png\' char(mouse_cd) char(day_cd) '_ep' num2str(EP1) '_vs_ep' num2str(EP2) 'activity' '.png'],'png')
    saveas(fig2,['J:\optodata\E186\221212\matlab_output\figures\svg\' char(mouse_cd) char(day_cd) '_ep' num2str(EP1) '_vs_ep' num2str(EP2) 'activity' '.svg'],'svg')
    catch
        mkdir('J:\optodata\E186\221212\matlab_output\figures\mat\')
        mkdir('J:\optodata\E186\221212\matlab_output\figures\png\')
        mkdir('J:\optodata\E186\221212\matlab_output\figures\svg\')

    saveas(fig,['J:\optodata\E186\221212\matlab_output\figures\mat\' char(mouse_cd) char(day_cd) '_ep' num2str(EP1) '_vs_ep' num2str(EP2)],'fig')
    saveas(fig,['J:\optodata\E186\221212\matlab_output\figures\png\' char(mouse_cd) char(day_cd) '_ep' num2str(EP1) '_vs_ep' num2str(EP2) '.png'],'png')
    saveas(fig,['J:\optodata\E186\221212\matlab_output\figures\svg\' char(mouse_cd) char(day_cd) '_ep' num2str(EP1) '_vs_ep' num2str(EP2) '.svg'],'svg')
    saveas(fig2,['J:\optodata\E186\221212\matlab_output\figures\mat\' char(mouse_cd) char(day_cd) '_ep' num2str(EP1) '_vs_ep' num2str(EP2) 'activity'],'fig')
    saveas(fig2,['J:\optodata\E186\221212\matlab_output\figures\png\' char(mouse_cd) char(day_cd) '_ep' num2str(EP1) '_vs_ep' num2str(EP2) 'activity' '.png'],'png')
    saveas(fig2,['J:\optodata\E186\221212\matlab_output\figures\svg\' char(mouse_cd) char(day_cd) '_ep' num2str(EP1) '_vs_ep' num2str(EP2) 'activity' '.svg'],'svg')
    end

end
