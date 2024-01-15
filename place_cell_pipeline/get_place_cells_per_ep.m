function [putative_pc] = get_place_cells_per_ep(eps,ep,ybinned,fv,Fc3,changeRewLoc, ...
    bin,track_length,gainf,Fs)
%%%%%%%% FC3 must be consistent in time x cell format %%%%%%%% 
thres = 5; % 5 cm/s is the velocity filter, only get
% frames when the animal is moving faster than that
ftol = 10; % number of frames length minimum to be considered stopped
eprng = eps(ep):eps(ep+1);
% mask = trialnum(eprng)>=3; % skip probes
ypos = ybinned(eprng);
ypos = ceil(ypos*(gainf)); % gain factor for zahra mice
fv = fv(eprng);
fc3 = Fc3(eprng,:);
% fc3 = fc3(:,mask);
rewloc = changeRewLoc(changeRewLoc>0); % reward location
rewlocep = rewloc(ep)*(gainf);

spatial_info = get_spatial_info_all_cells(fc3, fv,thres, ftol, ypos,Fs,...
    ceil(track_length/bin),track_length);    
% shuffle params
% shuffle transients and not fc3 values themselves
bins2shuffle_forcell = {};
for cshuf = 1:size(fc3,2)   
    bins2shuffle_forcell{cshuf} = shuffling_bins(fc3(:,cshuf))';   
end 
nshuffles = 1000;
spatial_info_shuf = zeros([nshuffles size(fc3,2)]);
shuffledbins_forcell(1:size(fc3,2)) = {0};
parfor j = 1:nshuffles
    disp(['Shuffle number ', num2str(j)]) 
    shuffled_cells_activity = zeros(size(fc3));
    shuffled_cells_activity = make_shuffled_Fc3_trace(bins2shuffle_forcell, ...
    fc3, shuffledbins_forcell, shuffled_cells_activity);
    spatial_info_shuf(j,:) = get_spatial_info_all_cells(shuffled_cells_activity, ...
        fv,thres, ftol,ypos,Fs,ceil(track_length/bin),track_length);
end
% compare spatial info to shuffled distribution
putative_pc = zeros(1,size(fc3,2)); % mask for pcs that pass shuffle crtieria
pvals = zeros(1,size(fc3,2)); % mask for pcs that pass shuffle crtieria
for cl=1:size(fc3,2)
    si = spatial_info(cl);
    si_shuf = sum(spatial_info_shuf(:,cl)>si)/nshuffles;
    pvals(cl) = si_shuf;
    if si_shuf<0.01 % 99 cut off
        putative_pc(cl) = 1;
    end
end    
putative_pc = logical(putative_pc);