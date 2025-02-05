function [opto, opto_comp, spike_opto, spike_opto_comp, ...
    spiketime_av_opto, spiketime_av_opto_comp] = split_neuro_data_with_opto_trials(n, ...
    eprng, eprng_comp,trialnum)
% n can be fc3 or bwlabel labels
neural_data = n(:,eprng);
mask = (trialnum(eprng)>=3) & (trialnum(eprng)<8);
opto = neural_data(:,mask);
%     comparison mask for first 5 trials or rest of ep
opto_comp = n(:, eprng_comp);
mask = trialnum(eprng_comp)>=8; %
opto_comp = opto_comp(:,mask);
opto_lbl = zeros(size(opto));
opto_lbl_comp = zeros(size(opto_comp));
% find number of spikes per cell
spike_opto = zeros(1,size(opto,1));
spike_opto_comp = zeros(1,size(opto_comp,1));
% find duration of spikes (aka how long fc3==0)
spiketime_av_opto = zeros(1,size(opto,1));
spiketime_av_opto_comp = zeros(1,size(opto_comp,1));
% average across time?
for i=1:size(opto_lbl)
    opto_lbl(i,:) = bwlabel(opto(i,:));
    spikes = unique(opto_lbl(i,:));
    spike_opto(i) = length(spikes)/size(opto_lbl,2);    
    for s=spikes
        if s ~= 0
            spiketime(s) = sum(opto_lbl(i,:)==spikes(s))/size(opto_lbl,2);
        end
    end
    spiketime_av_opto(i) = mean(spiketime);
    opto_lbl_comp(i,:) = bwlabel(opto_comp(i,:));
    spike_opto_comp(i) = sum(opto_lbl_comp(i,:)>0)/size(opto_lbl_comp,2);
    spikes = unique(opto_lbl_comp(i,:));
    for s=spikes
        if s ~= 0
            spiketime(s) = sum(opto_lbl(i,:)==spikes(s))/size(opto_lbl,2);
        end
    end
    spiketime_av_opto_comp(i) = mean(spiketime);
end

