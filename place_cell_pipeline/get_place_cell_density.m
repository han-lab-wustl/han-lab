function [pre,post,rew] = get_place_cell_density(eprng, dFF, fv, trialnum, pcs, track_length, rewloc, ybinned, Fs, ...
    window, nBins, thres, ftol)
%%%%% calcs place cell density pre, post, and at rew loc %%%%% 
first_trials = eprng(trialnum(eprng)>2 & trialnum(eprng)<6);
last_trials = eprng(trialnum(eprng)>max(trialnum(eprng))-3);
pcspop = sum(pcs,2)>1;
dff_first = dFF(first_trials, pcspop); % only place cells
dff_last = dFF(last_trials, pcspop); % only place cells
ypos_f = ybinned(first_trials);
ypos_l = ybinned(last_trials);
maskpre = (ypos_f<rewloc-window & ypos_f>30); % exclude first 30 cm
pref = get_spatial_sparsity_all_cells(dff_first(maskpre,:),...
    ypos_f(maskpre),nBins,track_length,fv(maskpre), thres, Fs, ftol);
maskpost = (ypos_f<rewloc+window & ypos_f<track_length-30); % also exclude last 30 cm
postf =get_spatial_sparsity_all_cells(dff_first(maskpost,:),...
    ypos_f(maskpost),nBins,track_length,fv(maskpost), thres, Fs, ftol);
rewf = get_spatial_sparsity_all_cells(dff_first((ypos_f>rewloc-window & ypos_f<rewloc+window),:),...
    ypos_f((ypos_f>rewloc-window & ypos_f<rewloc+window)),nBins,track_length,fv((ypos_f>rewloc-window & ypos_f<rewloc+window)), ...
    thres, Fs, ftol);
maskpre = (ypos_l<rewloc-window & ypos_l>30); % exclude first 30 cm
prel = get_spatial_sparsity_all_cells(dff_last(maskpre,:),...
    ypos_l(maskpre),nBins,track_length,fv(maskpre), thres, Fs, ftol);
maskpost = (ypos_l<rewloc+window & ypos_l<track_length-30); 
postl = get_spatial_sparsity_all_cells(dff_last(maskpost,:),...
    ypos_l(maskpost),nBins,track_length,fv(maskpost), thres, Fs, ftol);
rewl = get_spatial_sparsity_all_cells(dff_last((ypos_l>rewloc-window & ypos_l<rewloc+window),:),...
    ypos_l((ypos_l>rewloc-window & ypos_l<rewloc+window)),nBins,track_length,fv((ypos_l>rewloc-window & ypos_l<rewloc+window)), ...
    thres, Fs, ftol);
% pre=prel-pref;
% post=postl-postf;
% rew=rewl-rewf;
pre=mean(prel-pref, 'omitnan');
post=mean(postl-postf, 'omitnan');
rew=mean(rewl-rewf, 'omitnan');

end