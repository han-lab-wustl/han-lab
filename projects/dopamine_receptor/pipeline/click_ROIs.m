function click_ROIs(time, Fs, mat)
%% click_ROIs_MA
% select stabilized .mat video
% clip brightness lets you change the dynamic range to improve
% visualization
% first select how many well isolated cells you can select by hand
% afterwards you can add cells by manually drawing ROIs
close all
disp(fullfile(mat.folder, mat.name))
load(fullfile(mat.folder, mat.name));
inputdir = mat.folder; fn = mat.name;
if exist('chone_corr','var')
    video=chone_corr;
    clear chone_corr;
end
if exist('chone','var')
    video=chone;
    clear chone;
end
numframes=size(video,3);
window=round(Fs*time);
N=size(video,1);
M=size(video,2);
framestd=squeeze(std(single(video),0,3));
squeezeVid=reshape(video,size(video,1)*size(video,2),size(video,3));
framemean=squeeze(mean(single(video),3));
clear video;
frame=framestd;
tolerance = 300;

figure;
imagesc(framestd)
elide=input('Clip Brightness? (0 = no; 1 = yes): ');


if elide
    disp('Choose Brightest Pixel to Display')
    [x,y]=ginput(1);
    brtPix=framestd(round(y),round(x));
    disp('Choose Dimmest Pixel to Display')
    [x,y]=ginput(1);
    dimPix=framestd(round(y),round(x));
    framestd(framestd>brtPix)=brtPix;
    framestd(framestd<dimPix)=dimPix;
end

filt=fspecial('disk',20);
blurred=imfilter(framestd,filt,'replicate');
frame2=framestd./blurred;

rois=bwlabel(frame2>(mean(frame2(:))+1.5*std(frame2(:))),4);
rois=bwareaopen(rois,90);
rois=bwmorph(rois,'open');
rois=bwmorph(rois,'dilate');
rois1=double(rois);
rois=bwlabel(rois);

figure('units','normalized', 'Position', [.01 .05 .98 .87]);
[gx,gy]=gradient(rois1);
pic=zeros(size(frame2,1),size(frame2,2),3);
pic(:,:,1)=mat2gray(frame2);
pic(:,:,2)=mat2gray(frame2);
pic(:,:,3)=mat2gray(frame2)+mat2gray(squeeze(abs(gx)+abs(gy)));
imagesc(pic)

num_cells=1;
figure('units','normalized', 'Position', [.01 .05 .98 .87]);
imagesc(pic)
rois_chosen=zeros(num_cells,1);
inds_selected=rois_chosen;
masks=zeros(num_cells,size(rois,1),size(rois,2));
nega_masks=masks;
current_selection=0;
i = 0;
while 1
    i = i+1
    pts_selected(i,:)=ginput(1); %rows of [x,y]
    close all
    pts_selected(i,:)=round(pts_selected(i,:));
    inds_selected(i)=sub2ind(size(rois),pts_selected(i,2),pts_selected(i,1)); %takes in pts y,x. stupid
    
    pts_roi=find(rois>0);
    tem_roi=rois;
    rois_chosen(i)=rois(inds_selected(i));
    tem_roi(rois~=rois_chosen(i))=0;
    tem_roi(rois==rois_chosen(i))=1;
    if length(find(tem_roi)) >= size(rois,1)*size(rois,2)- length(find(rois)) - tolerance
          pts_selected(i,:) = [];
          inds_selected(i) = [];
          rois_chosen(i) = [];
        i = i-1;
        num_cells = num_cells-1;
         figure('units','normalized', 'Position', [.01 .05 .98 .87]);
    % hold on
    if size(masks,1)>1
        mframe=squeeze(sum(masks));
    else
        mframe=squeeze(masks);
    end
    pic(:,:,1)=mat2gray(frame2)+mat2gray(mframe);
    pic(:,:,2)=mat2gray(frame2);
    pic(:,:,3)=mat2gray(frame2)+mat2gray(squeeze(abs(gx)+abs(gy)));
    subplot(8,1,1:7)
    imagesc(pic)
    display(['Total cells = ' num2str(i)])
        break
    end
    [dx,dy]=gradient(double(tem_roi));
    borders=abs(dx)+abs(dy);
    tem_roi(borders>0)=1;
    nroi=tem_roi;
    for ii=1:20
        [dx,dy]=gradient(nroi);
        borders=abs(dx)+abs(dy);
        nroi(borders>0)=1;
    end
    nroi(rois>0)=0;
    masks(i,:,:)=tem_roi;
    nega_masks(i,:,:)=nroi;
    figure('units','normalized', 'Position', [.01 .05 .98 .87]);
    % hold on
    if size(masks,1)>1
        mframe=squeeze(sum(masks));
    else
        mframe=squeeze(masks);
    end
    pic(:,:,1)=mat2gray(frame2)+mat2gray(mframe);
    pic(:,:,2)=mat2gray(frame2);
    pic(:,:,3)=mat2gray(frame2)+mat2gray(squeeze(abs(gx)+abs(gy)));
    subplot(8,1,1:7)
    imagesc(pic)
    subplot(8,1,8)
    plot(constrain(squeeze(mean(squeezeVid(find(tem_roi),:)))));
    num_cells = num_cells+1;
end
frameout=frame2;
i=num_cells;

byhand=input('Add cells by hand? (0 = no; 1 = yes) ');
%%
if byhand
 [masks_hand,colors]=select_polys_GM(pic,45);
 for ii=1:size(masks_hand,1)
    nroi=masks_hand(ii,:,:);
    for jj=1:20
    [dx,dy]=gradient(double(nroi));
    borders=abs(dx)+abs(dy);
    nroi(borders>0)=1;
    end
    nroi(1,rois>0)=0;
    nega_masks(min(num_cells,i)+ii,:,:)=nroi;
 end
masks=[masks;masks_hand];
% rois=rois+squeeze(sum(masks_hand));
end
%now to find some neurites
%dilate rois
rois2=rois;
for ii=1:6
    [dx,dy]=gradient(rois2);
    borders=abs(dx)+abs(dy);
    rois2(borders>0)=1;
end
frame3=frame;
frame3(imclose(rois2,strel('disk',10))>0)=NaN;

% frame2=frame;
% frame2(rois>0)=NaN;


%% find dim and small neurons
figure('units','normalized', 'Position', [.01 .05 .98 .87]);
subplot(3,1,1)
imagesc(frame3);
colormap gray
freezeColors
subplot(3,1,2)
% filt=fspecial('disk',10);
% blurred=imfilter(frame2,filt,'replicate');
% frame2=frame2./blurred;
imagesc(frame2)
colormap jet
rois3=bwlabel(frame3>(nanmean(frame3(:))+.33*nanstd(frame3(:))),4);
initial_area=30;
rois3=bwareaopen(rois3,100);
rois3=bwlabel(rois3);
rois4=rois3;
n_loop=1;
while max(rois4(:))<5 && n_loop<20
    rois4=bwareaopen(rois3,round(initial_area*(1-.05*n_loop)));
    rois4=bwlabel(rois4);
    n_loop=n_loop+1;
end
n_loop=1;
while max(rois4(:)>30)
    rois4=bwareaopen(rois3,round(initial_area*(1+.1*n_loop)));
    n_loop=n_loop+1;
    rois4=bwlabel(rois4);
end
rois3=bwlabel(rois4);
subplot(3,1,3);
imagesc(rois3)
masks2=zeros(max(rois3(:))+1,size(rois,1),size(rois,2));
for j=1:max(rois3(:))
    tem_roi=rois3;
    tem_roi(rois3~=j)=0;
    tem_roi(rois3==j)=1;
    for jj=1:2
        [dx,dy]=gradient(double(tem_roi));
        borders=abs(dx)+abs(dy);
        tem_roi(borders>0)=1;
    end
tem_roi=imclose(tem_roi,strel('disk',2));
masks2(j,:,:)=tem_roi;

end
if size(masks,1)>1
    all_rois=sum(masks)+sum(masks2);
else
    all_rois=masks+sum(masks2);
end

masks2(end,:,:)=(all_rois==1);


%%
%Do this with contour eventually so it looks nicer
if size(masks,1)>1
    smasks=sum(masks(1:(end),:,:));
    snmasks=sum(nega_masks(1:(end),:,:));
else
    smasks=(masks(1,:,:));
    snmasks=(nega_masks(1,:,:));
end
[gx,gy]=gradient(double(smasks));
[gxn,gyn]=gradient(double(snmasks));

pic=zeros(size(masks,2),size(masks,3),3);
pic(:,:,1)=mat2gray(frame2)+mat2gray(squeeze(abs(gxn)+abs(gyn)));
pic(:,:,2)=mat2gray(frame2);
pic(:,:,3)=mat2gray(frame2)+mat2gray(squeeze(abs(gx)+abs(gy)));
%         imshow(mat2gray(100*frame/max(frame(:))+squeeze(10*gradient(sum(masks(1:(end-1),:,:),1)))));
figure
imagesc(pic)
saveas(gcf,[inputdir,fn(1:(end-4)),'frame.jpg'])
figure
% pic2=insertText(pic,find_centroids(masks),1:(size(masks,1)),'FontSize', 8, 'BoxColor', 'White', 'BoxOpacity', 0, 'AnchorPoint', 'Center','TextColor',[.1 .8 .1]);
imagesc(pic)
    cents=find_centroids(masks);
for ii=1:size(masks,1)
       text(cents(ii,1),cents(ii,2),num2str(ii),'Color','w','HorizontalAlignment','Center')
end
saveas(gcf,[inputdir,fn(1:(end-4)),'masks.jpg'])
figure
indmasks=bsxfun(@times,masks,(1:size(masks,1))');
imagesc(squeeze(sum(indmasks)));
saveas(gcf,[inputdir,fn(1:(end-4)),'blotches.jpg'])

video=reshape(squeezeVid,M,N,numframes);
clear squeezeVid
[F,nF,FminusN]=extract_F_poly_from_auto_ROI(masks,video,nega_masks);
Fc=zeros(size(F));
dFF=zeros(size(F));
% Fs=zeros(size(F));
for j=1:size(F,2)
    junk=FminusN(:,j);
    
%     window=round(numframes/numwindow);
    junk2=zeros(size(junk));
    for k=1:length(junk)
        cut=junk(max(1,k-window):min(numframes,k+window));
        cutsort=sort(cut);
        a=round(length(cut)*.08);
        junk2(k)=cutsort(a);
    end
    Fc(:,j)=(junk./junk2);
    maxval=max(Fc(:,j));
    Fc(:,j)=(Fc(:,j)-1)/max((Fc(:,j)-1));
    Fc(:,j)=maxval*Fc(:,j);
    dFF(:,j)=(junk-junk2)./junk2;
    F0=mean(junk2);
    %Fc(:,i)=(junk-junk2)-mean((junk-junk2));
end

frame=frameout;

fullFname=[inputdir,fn(1:(end-4)),'_roibyclick_F.mat'];
Frawt=F;
save(fullFname,'F','Frawt','nF','Fc','F0','dFF','masks','numframes','N','M','window','nega_masks','frame');


figure('units','normalized', 'Position', [.01 .05 .98 .87]);
    subplot(1,2,1)
    masks(masks>0)=1;
    [gx1,gy1]=gradient(sum(masks(:,:,:),1));
    pic=zeros(size(masks,2),size(masks,3),3);
    frame3=framestd./blurred;
    pic(:,:,1)=mat2gray(frame3);
    pic(:,:,2)=mat2gray(frame3);
    pic(:,:,3)=mat2gray(frame3)+mat2gray(squeeze(abs(gx1)+abs(gy1)));
        cents=find_centroids(masks);

    imagesc(pic)
        for ii=1:size(masks,1)
       text(cents(ii,1),cents(ii,2),num2str(ii),'Color','w','HorizontalAlignment','Center')
        end
    subplot(1,2,2)
    plot(bsxfun(@plus,dFF,1:size(dFF,2)))
    saveas(gcf,[inputdir,fn(1:(end-4)),'RoiFcs.jpg'])

end