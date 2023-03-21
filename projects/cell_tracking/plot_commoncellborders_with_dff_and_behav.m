% make an array of frame, cell border, and trace per cell below (multiplot) and export as tif
%test
border='Z:/20230124_run/cell_borders_per_session/221206_cellno00013.tif';
frames='H:/imaging_data/221206/suite2p/plane0/reg_tif/file000_chan0.tif';
behavior='Z:/cellregtest_Fmats/221206_Fall.mat';
dFF = 'Z:/dff_221206-11.mat';
% need to load commoncells var too
dff = load(dFF);
b=imread(border);
t=tiffreadVolume(frames);
behav=load(behavior);
% plot cell border over image
frameno=1;
f=figure;
subplot(2,1,1);
im = imoverlay(t(:,:,frameno),b,'yellow');
imshow(im); colormap('gray')
subplot(2,1,2)
plot(behav.ybinned, 'Color', grayColor); hold on; 
plot(behav.changeRewLoc, 'b')
plot(find(behav.licks),behav.ybinned(find(behav.licks)),'r.') % what is this doing?
yyaxis right
dfff=dff.dff{1}(commoncells(14,1),:);
plot(dfff,'g') %cell 14,day1
plot(frameno,dfff(frameno),'ko','MarkerSize',10') %max df at that day
saveas(f,'Z:/test.tif')


imagesc(t(:,:,1))
colormap('gray')
hold on;
imagesc(b);
colormap('gray');
hold off;