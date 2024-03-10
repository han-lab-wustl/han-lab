% zahra
% check fov per day
% for alignment across days
clear all; close all
src = 'X:\vipcre\e216';
dys = dir(src); dys=dys(~ismember({dys.name},{'.','..'}));
dys_nm = arrayfun(@(x) str2double(x.name), dys);
[~,idx] = sort(dys_nm);
dys_nm = dys_nm(idx);
fmats = dir(fullfile(src, '**\Fall.mat'));
fmats = fmats(idx);
figure;
for f=1:length(fmats)
    subplot(ceil(sqrt(length(fmats))), ceil(sqrt(length(fmats))), f)
    load(fullfile(fmats(f).folder, fmats(f).name), 'ops')
    imagesc(ops.meanImg)
    title(dys_nm(f))
end