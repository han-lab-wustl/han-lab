% grab cell # in its tuning curve across days
% use output from Ele's masterhrz script (modified by Zahra)
% REMEMBER THAT THESE ARE THE ONLY 107 CELLS TRACKED WITH CELLREG ACROSS
% WEEKS AND THEN MAPPED TO DAYS WITH CELLREG

clear all;
fls = dir('Z:\HRZ_master_output8trials\*.mat');
sessions=length(fls);
plot_or_not = 0; %individual plots of tuning curve per day
% load mats from all days
fls2 = dir(fullfile('Z:\cellreg1month_Fmats\', '**\*YC_Fall.mat'));%dir('Z:\cellreg1month_Fmats\*YC_Fall.mat');
days = cell(1, length(fls2));
for fl=1:length(fls2)
    day = fls2(fl);
    days{fl} = load(fullfile(day.folder,day.name));
end

%%
clear TC_imagesc_arr comp_arr day_arr

for cellno=1:107 %hard coded cell no.
    for day=1:sessions % grab across days
        load(fullfile(fls(day).folder, fls(day).name))
        fl=fullfile(fls(day).folder, fls(day).name);
        day_cd=fl(end-28:end-25); % hack lol
        comparisons = size(cs_table,1);
        for this_comparison=1:comparisons
            % imagesc plot
            %sorts by the maximum firing rate of the each cell
            try 
                [~,max_bin1] = max(cs_table.tuning_curves_for_this_comparison{this_comparison,1},[],2);
                [~,max_bin2] = max(cs_table.tuning_curves_for_this_comparison{this_comparison,2},[],2);
                [~,sorted_idx] = sort(max_bin1);
                
                TC_imagesc = [cs_table.tuning_curves_for_this_comparison{this_comparison,1}(sorted_idx,:) ...
                    cs_table.tuning_curves_for_this_comparison{this_comparison,2}(sorted_idx,:)];
            catch %wont work for days with only 1 epoch...
                TC_imagesc = [cs_table.tuning_curves{1}(sorted_idx,:) cs_table.tuning_curves{1}(sorted_idx,:)];
            end
            % want to collect this across epochs and days...
            TC_imagesc=TC_imagesc(sorted_idx==cellno,:);
            if plot_or_not
                figure;
                imagesc(normalize(TC_imagesc,2,'range'))%imagesc(normalize(TC_imagesc,2,'range'))
                colormap(turbo)
                xlabel('cm')
                xticks([0 max([max_bin1; max_bin2])])
                xticklabels([0 180])
                yticks([1])
                yticklabels(cellno) %label with cell id number
                a = get(gca,'XTickLabel');
                set(gca,'XTickLabel',a,'FontName','Times','fontsize',15)
                title(['Day=' day_cd '; TC last 8 trials'],{['epoch ' ...
                    num2str(cs_table.comparison_type(this_comparison,1)) ';epoch ' num2str(cs_table.comparison_type(this_comparison,2))]})                      
            end
            if ~sum(TC_imagesc)==0
                TC_imagesc_arr(cellno,day,this_comparison,:) = TC_imagesc;
                comp_arr(cellno,day,this_comparison,:) = [cs_table.comparison_type(this_comparison,1) cs_table.comparison_type(this_comparison,2)];
                day_arr{cellno,day}=day_cd;
            end            
        end
        % how to find location of epochs
        %eg
        l = days{1,day};
        first_rewarded_trial = find(l.trialnum >= 3,1) ;
        single_trials_boundaries = [first_rewarded_trial diff(l.trialnum) == 1];
        first_epoch_trials = l.trialnum == 3; %first epoch is always 3
        if l.trialnum(1)>3 %unless when it is not..
            first_epoch_trials(1) = l.trialnum(1)>3;
        end
        epoch_LLs = find(single_trials_boundaries & first_epoch_trials);
        last_trial_is_a_probe_trial = l.trialnum(length_recording) < 3;
        if last_trial_is_a_probe_trial
            epoch_ULs = find(diff(l.trialnum)< -3);
        else
            epoch_ULs = [find(diff(l.trialnum)< -3) length_recording];
        end
        epoch_ULs_arr{day}=epoch_ULs; %epoch location based on frames
    end
end
%%
close all

%test
%reformat for plotting
%make reward zone matrix by multiplying by a factor (the 'bin')
%this is a hack
for i=1:sessions
    day_arr_re(i) = string(day_arr{1,i});
end

for cellno=1:107  
    plt=squeeze(TC_imagesc_arr(cellno,:,1,:)); %only looking at first comparison
    figure(cellno);
    %reformat for plotting
    imagesc(normalize(plt,2,'range'));yticks(1:length(day_arr));yticklabels(day_arr_re)
    xlabel('cm')
    xticks([0 54 108])
    xticklabels([0 54 54])
    patch([min(xlim) max(xlim) max(xlim) min(xlim)], [ybars(1) ybars(1), ybars(2) ybars(2)], [0.8 0.8 0.8])
end