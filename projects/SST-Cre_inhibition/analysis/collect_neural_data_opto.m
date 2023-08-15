function [dffs,diff_opto,spatial_info,spike_optos,spike_opto_comps,...
    spiketime_av_optos, spiketime_av_opto_comps] = collect_neural_data_opto(days,srcdir,animal,ep, ...
    num_trials,plot_dff)
% zd's function for gathering neural data
spike_optos = {};
spike_opto_comps = {};
opto_means = {};
opto_comp_means = {};
spiketime_av_optos = {};
spiketime_av_opto_comps = {};

%   Detailed explanation goes here
for d=1:length(days)
    disp(days(d))
    clearvars -except days srcdir animal ep num_trials ...
    plot_dff d spike_optos opto_means opto_comp_means spike_opto_comps spiketime_av_optos ...
    spiketime_av_opto_comps
    fmatfl = dir(fullfile(srcdir, animal, string(days(d)), "**\Fall.mat")); 
    load(fullfile(fmatfl.folder,fmatfl.name));
    Fall = load(fullfile(fmatfl.folder,fmatfl.name)); % to avoid iscell conflitcts
    
    eps = find(changeRewLoc);
    eps = [eps length(changeRewLoc)]; % includes end of recording as end of a epoch
    % ep 2
    if ep==1
        eprng = eps(1):eps(2);
    elseif ep==2
        eprng = eps(2):eps(3);
    elseif ep==3        
        eprng = eps(3):eps(4);        
    end
    rewloc = changeRewLoc(changeRewLoc>0); % reward location
    rewlocopto = rewloc(ep);
    prevrewlocopto = rewloc(ep-1);
    % set reward zone
    rewzoneopto = find_reward_zone(rewlocopto);
    prevrewzoneopto = find_reward_zone(prevrewlocopto);
%     eprng_comp_m = ones(1,length(changeRewLoc));%eps(1):eps(2);
%     eprng_comp_m(eprng) = 0; % remove opto ep but keep the rest
%     rng = [1:length(changeRewLoc)];
    eprng_comp = eprng;%rng(logical(eprng_comp_m));
    [dff,bordercells] = remove_border_cells(stat, Fall.iscell, F, ...
    cell2remove, remove_iscell, Fall.all);
    fc3 = all.Fc3(~logical(bordercells),:);
    % optional!! skewness filter
%     [dff, fc3] = skewness_filter(dff, fc3); dffs{d} = dff;    
    % #1 find spatial info and add to Fall
    % 10 cm bins
    bin = 10; % cm t
    track_length = 270;
    spatial_info{d} = get_spatial_info_all_cells(dff',ybinned,31.25, ...
        ceil(track_length/bin),track_length);
    save(fullfile(fmatfl.folder,fmatfl.name),'spatial_info','-append')
    
    [opto, opto_comp, spike_opto, spike_opto_comp, ...
    spiketime_av_opto, spiketime_av_opto_comp] = split_neuro_data_with_opto_trials(fc3, ...
    eprng, eprng_comp, trialnum);
    % take mean across time
    opto_mean = mean(opto, 2);
    opto_comp_mean = mean(opto_comp, 2);
    spike_optos{d} = spike_opto;
    spike_opto_comps{d} = spike_opto_comp;
    spiketime_av_optos{d} = spiketime_av_opto;
    spiketime_av_opto_comps{d} = spiketime_av_opto_comp;
    opto_means{d} = opto_mean;
    opto_comp_means{d} = opto_comp_mean;
    % vars for plotting
    if plot_dff==1
        data_ = mean(fc3,1,'omitnan');%smoothdata(mean(dff,1,'omitnan'), 'gaussian',20);
        figure; subplot(3,1,1);
        plot(data_,'k'); hold on;
        for mm = 1:length(eps)-1 %the rectangle indicating the reward location, overlaps the probe trials referring to the previous reward location
            rectangle('position',[eps(mm) min(data_) length(find(trialnum(eps(mm):eps(mm+1)-1)<3)) max(data_)-min(data_)], ... % just picked max for visualization
                'EdgeColor',[0 0 0 0],'FaceColor',[0 .5 .5 0.3])
        end
        optoep = eprng;
        rectangle('position',[min(optoep(trialnum(optoep)>=3)) min(data_) ...
            length(optoep((trialnum(optoep)>=3) & (trialnum(optoep)<3+num_trials))) max(data_)-min(data_)], ...
                'EdgeColor',[0 0 0 0],'FaceColor',[1 0 0 0.3])        
        
        subplot(3,1,2)
        plot(forwardvel, 'b'); xlim([0 length(ybinned)]); hold on;
        for mm = 1:length(eps)-1 %the rectangle indicating the reward location, overlaps the probe trials referring to the previous reward location
            rectangle('position',[eps(mm) min(data_) length(find(trialnum(eps(mm):eps(mm+1)-1)<3)) max(data_)-min(data_)], ... % just picked max for visualization
                'EdgeColor',[0 0 0 0],'FaceColor',[0 .5 .5 0.3])
        end
        subplot(3,1,3); 
        plot(ybinned, 'k'); xlim([0 length(ybinned)])
        hold on; 
        plot(find(rewards==1), ybinned(rewards==1), 'go')
        plot(find(licks), ybinned(licks), 'r.')
        for mm = 1:length(eps)-1 %the rectangle indicating the reward location, overlaps the probe trials referring to the previous reward location
            rectangle('position',[eps(mm) min(data_) length(find(trialnum(eps(mm):eps(mm+1)-1)<3)) max(data_)-min(data_)], ... % just picked max for visualization
                'EdgeColor',[0 0 0 0],'FaceColor',[0 .5 .5 0.3])
        end
        sgtitle(sprintf('animal = %s, day = %i, ep = %i \n rewzone = %i to %i', animal,days(d), ...
            ep, prevrewzoneopto, rewzoneopto))
    end
end

diff_opto = diff([cell2mat(cellfun(@(x) x', opto_means, 'UniformOutput', false)); ...
    cell2mat(cellfun(@(x) x', opto_comp_means, 'UniformOutput', false))]);
end