% zahra
% check fov per day
% for alignment across days
clear all; close all
src = 'X:\vipcre\e216';
fmats = dir(fullfile(src, '**\Fall.mat'));
figure;
for f=1:length(fmats)
    subplot(ceil(sqrt(length(fmats))), ceil(sqrt(length(fmats))), f)
    load(fullfile(fmats(f).folder, fmats(f).name), 'ops')
    imagesc(ops.meanImg)
    title(sprintf('day %i', f))
end