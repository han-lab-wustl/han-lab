% select an open fig
% then select the subplot you want to copy/pop out
set(gca,'Renderer','Painters')
fig1 = get(gca,'Children');
figure;
axes;
copyobj(fig1,gca);
colormap(bone)
set(gca,'Ydir','reverse')
axis tight