% plot PCs before and after opto periods
clear all; clear all; close all;
srcdir = 'Z:\sstcre_imaging';
animal = 'e201';
optodays = [55,58,61,64,67];
for d=optodays
fmatfl = dir(fullfile(srcdir, animal, string(d), "**\Fall.mat")); 
load(fullfile(fmatfl.folder,fmatfl.name));
Fall = load(fullfile(fmatfl.folder,fmatfl.name));
eps = find(changeRewLoc);
eps = [eps length(changeRewLoc)]; % includes end of recording as end of a epocheprngbefore = eps(1):eps(2);
eprngopto = eps(2):eps(3);
eprngbefore = eps(1):eps(2);
optomask = (trialnum(eprngopto)<15);
beforemask = (trialnum(eprngbefore)>=max(trialnum(eprngbefore))-10);
rng = [eprngbefore(beforemask) eprngopto(optomask)];
dff = remove_border_cells(stat, Fall.iscell, F, ...
    cell2remove, remove_iscell, all);
dff = skewness_filter(dff);
cells = 100;
% add window of ypos for averaging activity around reward
rewloc = changeRewLoc(changeRewLoc>0); % reward location
rewlocopto = rewloc(2);
% rng = rng((ybinned(rng)>=rewlocopto-10)&(ybinned(rng)<=rewlocopto+5));
%%
figure;
plotrng = length(rng);
subplot(4,1,1)
imagesc(dff(1:cells,rng)); hold on;
%probes
rectangle('position',[find(trialnum(rng)<3, 1) 0 ...
    length(find(trialnum(rng)<3)) cells], ... % just picked max for visualization
                'EdgeColor',[0 0 0 0],'FaceColor',[0 .5 .5 0.3])
%opto period
rectangle('position',[min(find((trialnum(rng)>=3)&(trialnum(rng)<8))) 0 ...
            length(find((trialnum(rng)>=3)&(trialnum(rng)<8))) cells], ...
                'EdgeColor',[0 0 0 0],'FaceColor',[1 0 0 0.3]) 
subplot(4,1,2)
% plot(dff(randi([0 size(dff,1)],1),rng),'g'); xlim([0 plotrng]);hold on;
% plot(mean(dff(:,rng),1),'g'); xlim([0 plotrng]);hold on;
plot(mean(dff(:,rng),1),'g'); xlim([0 plotrng]);hold on;
%probes
rectangle('position',[find(trialnum(rng)<3, 1) min(mean(dff(:,rng),1)) ...
    length(find(trialnum(rng)<3)) max(mean(dff(:,rng),1))-min(mean(dff(:,rng),1))], ... % just picked max for visualization
                'EdgeColor',[0 0 0 0],'FaceColor',[0 .5 .5 0.3])
%opto period
rectangle('position',[min(find((trialnum(rng)>=3)&(trialnum(rng)<8))) min(mean(dff(:,rng),1)) ...
            length(find((trialnum(rng)>=3)&(trialnum(rng)<8))) max(mean(dff(:,rng),1))-min(mean(dff(:,rng),1))], ...
                'EdgeColor',[0 0 0 0],'FaceColor',[1 0 0 0.3]) 
subplot(4,1,3)
plot(forwardvel(rng), 'b'); xlim([0 plotrng]); hold on;
%probes
rectangle('position',[find(trialnum(rng)<3, 1) min(forwardvel(rng)) ...
    length(find(trialnum(rng)<3)) max(forwardvel(rng))-min(forwardvel(rng))], ... % just picked max for visualization
                'EdgeColor',[0 0 0 0],'FaceColor',[0 .5 .5 0.3])
%opto period
rectangle('position',[min(find((trialnum(rng)>=3)&(trialnum(rng)<8))) min(forwardvel(rng)) ...
            length(find((trialnum(rng)>=3)&(trialnum(rng)<8))) max(forwardvel(rng))-min(forwardvel(rng))], ...
                'EdgeColor',[0 0 0 0],'FaceColor',[1 0 0 0.3]) 
subplot(4,1,4); 
plot(ybinned(rng), 'k'); xlim([0 plotrng])
hold on; 
ybin = ybinned(rng);
plot(find(rewards(rng)==1), ybin(rewards(rng)==1), 'go')
plot(find(licks(rng)), ybin(licks(rng)), 'r.')
%probes
rectangle('position',[find(trialnum(rng)<3, 1) min(ybinned(rng)) ...
    length(find(trialnum(rng)<3)) max(ybinned(rng))-min(ybinned(rng))], ... % just picked max for visualization
                'EdgeColor',[0 0 0 0],'FaceColor',[0 .5 .5 0.3])
%opto period
rectangle('position',[min(find((trialnum(rng)>=3)&(trialnum(rng)<8))) min(ybinned(rng)) ...
            length(find((trialnum(rng)>=3)&(trialnum(rng)<8))) max(ybinned(rng))-min(ybinned(rng))], ...
                'EdgeColor',[0 0 0 0],'FaceColor',[1 0 0 0.3]) 
sgtitle(sprintf('opto, e201, day %i', d))
%%
end
%%
clear all; clear all; 
srcdir = 'Z:\sstcre_imaging';
animal = 'e201';
ctrldays = [57,60,63,66,69];

for d=ctrldays
fmatfl = dir(fullfile(srcdir, animal, string(d), "**\Fall.mat")); 
load(fullfile(fmatfl.folder,fmatfl.name));
Fall = load(fullfile(fmatfl.folder,fmatfl.name));
eps = find(changeRewLoc);
eps = [eps length(changeRewLoc)]; % includes end of recording as end of a epocheprngbefore = eps(1):eps(2);
eprngopto = eps(2):eps(3);
eprngbefore = eps(1):eps(2);
optomask = (trialnum(eprngopto)<15);
beforemask = (trialnum(eprngbefore)>=max(trialnum(eprngbefore))-5);
rng = [eprngbefore(beforemask) eprngopto(optomask)];
dff = remove_border_cells(stat, Fall.iscell, F, ...
    cell2remove, remove_iscell, all);
dff = skewness_filter(dff);
cells = 100;
% add window of ypos for averaging activity around reward
rewloc = changeRewLoc(changeRewLoc>0); % reward location
rewlocopto = rewloc(2);
rng = rng((ybinned(rng)>rewlocopto-10)&(ybinned(rng)<=rewlocopto+5));

figure;
plotrng = length(rng);
subplot(4,1,1)
imagesc(dff(1:cells,rng));hold on;
%probes
rectangle('position',[find(trialnum(rng)<3, 1) 0 ...
    length(find(trialnum(rng)<3)) cells], ... % just picked max for visualization
                'EdgeColor',[0 0 0 0],'FaceColor',[0 .5 .5 0.3])
%opto period
rectangle('position',[min(find((trialnum(rng)>=3)&(trialnum(rng)<8))) 0 ...
            length(find((trialnum(rng)>=3)&(trialnum(rng)<8))) cells], ...
                'EdgeColor',[0 0 0 0],'FaceColor',[1 0 0 0.3]) 
subplot(4,1,2)
plot(mean(dff(:,rng),1),'g'); xlim([0 plotrng]);hold on;
%probes
rectangle('position',[find(trialnum(rng)<3, 1) min(mean(dff(:,rng),1)) ...
    length(find(trialnum(rng)<3)) max(mean(dff(:,rng),1))-min(mean(dff(:,rng),1))], ... % just picked max for visualization
                'EdgeColor',[0 0 0 0],'FaceColor',[0 .5 .5 0.3])
%opto period
rectangle('position',[min(find((trialnum(rng)>=3)&(trialnum(rng)<8))) min(mean(dff(:,rng),1)) ...
            length(find((trialnum(rng)>=3)&(trialnum(rng)<8))) max(mean(dff(:,rng),1))-min(mean(dff(:,rng),1))], ...
                'EdgeColor',[0 0 0 0],'FaceColor',[1 0 0 0.3]) 
subplot(4,1,3)
plot(forwardvel(rng), 'b'); xlim([0 plotrng]); hold on;
%probes
rectangle('position',[find(trialnum(rng)<3, 1) min(forwardvel(rng)) ...
    length(find(trialnum(rng)<3)) max(forwardvel(rng))-min(forwardvel(rng))], ... % just picked max for visualization
                'EdgeColor',[0 0 0 0],'FaceColor',[0 .5 .5 0.3])
%opto period
rectangle('position',[min(find((trialnum(rng)>=3)&(trialnum(rng)<8))) min(forwardvel(rng)) ...
            length(find((trialnum(rng)>=3)&(trialnum(rng)<8))) max(forwardvel(rng))-min(forwardvel(rng))], ...
                'EdgeColor',[0 0 0 0],'FaceColor',[1 0 0 0.3]) 
subplot(4,1,4); 
plot(ybinned(rng), 'k'); xlim([0 plotrng])
hold on; 
ybin = ybinned(rng);
plot(find(rewards(rng)==1), ybin(rewards(rng)==1), 'go')
plot(find(licks(rng)), ybin(licks(rng)), 'r.')
%probes
rectangle('position',[find(trialnum(rng)<3, 1) min(ybinned(rng)) ...
    length(find(trialnum(rng)<3)) max(ybinned(rng))-min(ybinned(rng))], ... % just picked max for visualization
                'EdgeColor',[0 0 0 0],'FaceColor',[0 .5 .5 0.3])
%opto period
rectangle('position',[min(find((trialnum(rng)>=3)&(trialnum(rng)<8))) min(ybinned(rng)) ...
            length(find((trialnum(rng)>=3)&(trialnum(rng)<8))) max(ybinned(rng))-min(ybinned(rng))], ...
                'EdgeColor',[0 0 0 0],'FaceColor',[1 0 0 0.3]) 
sgtitle(sprintf('ctrl, e201, day %i', d))
end