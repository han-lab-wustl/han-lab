function [masks,colors]=select_polys_GM(imgdata,fignum)

 figure('units','normalized', 'Position', [.01 .05 .98 .89]);
 clf;
 colormap(gray);
 img1 = imagesc(imgdata(:,:,2));
%  imagesc(imgdata);
 [xx,yy]=meshgrid(1:size(imgdata,2),1:size(imgdata,1));
 set(gcf,'DefaultTextColor','blue')
 set(gcf,'DefaultTextFontSize',18);
i=0;
colors=[];
addmore = 1;
while addmore
    i=i+1;
    [masks(i,:,:),xi,yi]=roipoly;
    colors=add_random_color(colors,.4);
    for j=1:length(xi)-1
        L1=line([xi(j) xi(j+1)],[yi(j) yi(j+1)]);
        set(L1,'Color',colors(:,i));     
        set(L1,'LineWidth',2);
    end
    
    thismask=find(squeeze(masks(i,:,:))==1);
    H=text(median(xx(thismask)),median(yy(thismask)),sprintf('%d',i));
    set(H,'HorizontalAlignment','center');

  %  plot(min(X(:,i)),min(Y(:,i)),'ro');
   % plot(max(X(:,i)),max(Y(:,i)),'go');
   addmore = input('Add More? (0 - no 1 - yes) ');
end
masks=masks(1:i,:,:);