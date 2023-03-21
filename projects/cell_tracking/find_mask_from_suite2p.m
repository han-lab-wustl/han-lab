% by Zahra
% plot mask of detected cells in suite2p and save as mat
mat = load('H:\imaging_data\221206\suite2p\plane0\Fall.mat');
figure; imagesc(mean(allFiltersMat,3))%imagesc(mat.ops.meanImg);
colormap('gray')
hold on;
celln=size(mat.stat);celln=celln(2);
for i=1:celln
    if iscell(i,1) %only plot what 2p recognizes as cells
        plot(mat.stat{1,i}.xpix, mat.stat{1,i}.ypix, 'r')
        hold on;
    end
end