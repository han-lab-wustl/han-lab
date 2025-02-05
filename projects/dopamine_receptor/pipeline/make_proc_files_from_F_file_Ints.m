function make_proc_files_from_F_file_Ints(plane,day_num,fclick, fall)

day_1 = load(fullfile(fclick.folder,fclick.name));
dat = load(fullfile(fall.folder,fall.name));

% zd added 
nCells = size(day_1.dFF,2);
del_idx = [];
for i = 1:length(dat.stat)
    if i <= nCells
        dat.stat{i}.iscell = 1;
    else
        dat.stat{i}.iscell = 0;
        del_idx = [del_idx i];
    end
end

dat.stat(del_idx) = [];
%%
% mimg in dat struct
dat.mimg = [];
alpha = zeros(size(day_1.frame));
dat.mimg(:,:,1) = alpha; 
dat.mimg(:,:,2) = day_1.frame;
dat.mimg(:,:,3) = alpha;
dat.mimg(:,:,4) = alpha;
dat.mimg(:,:,5) = day_1.frame;

% mimg_proc in dat_struct
dat.mimg_proc = [];
alpha = zeros(size(day_1.frame));
dat.mimg_proc(:,:,1) = alpha; 
dat.mimg_proc(:,:,2) = day_1.frame;
dat.mimg_proc(:,:,3) = alpha;
dat.mimg_proc(:,:,4) = alpha;
dat.mimg_proc(:,:,5) = day_1.frame;

%xlim in dat struct
dat.xlim(1) = 0; 
dat.xlim(2) = size(day_1.frame,2);

%ylim in dat struct
dat.ylim(1) = 0; 
dat.ylim(2) = size(day_1.frame,1);

%dat.cl
dat.cl.Lx = size(day_1.frame,2);
dat.cl.Ly = size(day_1.frame,1);

%dat.res
dat.res.lambda = [];
dat.res.lambda0 = [];
dat.res.lambda = day_1.frame; 
dat.res.lambda0 = day_1.frame;
dat.res.iclust = [];
dat.res.iclust = zeros(size(day_1.frame));
for i = 1:nCells
    dat.res.iclust(squeeze(day_1.masks(i,:,:)) == 1) = i;
end


%%
% dat.stat.xpix

for i = 1:nCells
    mask = squeeze(day_1.masks(i,:,:));
    cell_x = [];
    cell_y = [];
    for ii = 1:size(mask,1)
        for jj = 1:size(mask,2)
            if mask(ii,jj) == 1
                cell_x = [cell_x,jj];
                cell_y = [cell_y,ii];
            end
        end
    end
    dat.stat{i}.xpix = [];
    dat.stat{i}.xpix = cell_x;
    dat.stat{i}.ypix = [];
    dat.stat{i}.ypix = cell_y;
    dat.stat{i}.ipix = [];
    dat.stat{i}.ipix = cell_x; 
    dat.stat{i}.lam = [];
    dat.stat{i}.lam = ones(length(cell_x),1);
    dat.stat{i}.lambda = [];
    dat.stat{i}.lambda = ones(length(cell_x),1);
    dat.stat{i}.npix = length(cell_x);
    dat.stat{i}.med(1) = median(cell_x);
    dat.stat{i}.med(2) = median(cell_y);
    clear cell_x cell_y mask
end

%%
for i = 1:length(dat.stat)
    iscell(i) = dat.iscell(i,1);
end

% dat.F.ichose
dat.ichosen = sum(iscell);
%%

% dat.ops.mimg1

dat.ops.mimg1 = day_1.frame;
dat.ops.sdmov = day_1.frame;
dat.ops.Vcorr = day_1.frame;
dat.ops.xrange = 1:size(day_1.frame,2);
dat.ops.yrange = (1:size(day_1.frame,1))';

filename = [fclick.name(1:end-4) '_int_proc_file.mat'];
save(fullfile(fclick.folder, filename) ,'dat');
