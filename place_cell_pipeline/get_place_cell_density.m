function [pre,post,rew] = get_place_cell_density(eprng, dFF, trialnum, pcs, coms, rewloc, ybinned, ep)
first_trials = eprng(trialnum(eprng)>2 & trialnum(eprng)<8);
last_trials = eprng(trialnum(eprng)>max(trialnum(eprng))-5);
pcspop = any(pcs,2);
dff_first = dFF(first_trials, pcspop); % only place cells
dff_last = dFF(last_trials, pcspop); % only place cells
ypos_f = ybinned(first_trials);
ypos_l = ybinned(last_trials);
dff_com_f = [];
dff_com_l = [];
com_ep = coms{ep};
com_ep(isnan(com_ep))=1; % temp
for pc=1:sum(pcspop)
dff_com_f(pc) = max(dff_first((ypos_f>com_ep(pc)-10 & ypos_f<com_ep(pc)+10),pc));%, 'omitnan'); % 5 cm window around com
dff_com_l(pc) = max(dff_last((ypos_l>com_ep(pc)-10 & ypos_l<com_ep(pc)+10),pc));%, 'omitnan');
end
window = 10; 
pref = mean(dff_com_f(com_ep<rewloc-window), 'omitnan'); % 10 cm window
postf = mean(dff_com_f(com_ep>rewloc+window), 'omitnan');
rewf = mean(dff_com_f(com_ep>rewloc-window & com_ep<rewloc+window), 'omitnan');
prel = mean(dff_com_l(com_ep<rewloc-window), 'omitnan'); % 10 cm window
postl = mean(dff_com_l(com_ep>rewloc+window), 'omitnan');
rewl = mean(dff_com_l(com_ep>rewloc-window & com_ep<rewloc+window), 'omitnan');
pre=prel-pref;
post=postl-postf;
rew=rewl-rewf;
end