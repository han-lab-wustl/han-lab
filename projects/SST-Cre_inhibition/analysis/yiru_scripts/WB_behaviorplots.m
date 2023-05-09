%%
dFF=all.dff;
%%
for currentcell = 29 %For cell n, choose currencell= n+1
figure;
scatter(timedFF,ybinned,8,[0.7 0.7 0.7],'filled')
hold on
scatter(timedFF(find(licks)),ybinned(find(licks)),8,'yx')
scatter(timedFF(find(rewards)),ybinned(find(rewards)),8,'r','filled')

plot(timedFF,rescale(dFF(currentcell, :),0,180),'Color',[0 0 0.5])
title(['Cell ' num2str(currentcell)])
% if skewness(dFF(currentcell, :), 0, 'all') < 2:
%      savefig(['J:\E186_plots\DffAndBeh\221109\2_IN\Cell' num2str(currentcell)])
% else if skewness(dFF(currentcell, :), 0, 'all') >2:
%          savefig(['J:\E186_plots\DffAndBeh\221109\1_pyramidal\Cell' num2str(currentcell)])
%savefig(['J:\E186_plots\DffAndBeh\221119\Cell' num2str(currentcell)])
pause(0.8)
%close 
hold off
end
%%

figure;
for currentcell = 1:184
range = 10;
binsize = 0.1;
fakerewards = rewards == 0.5;
[meany,x,y] = perirewardbinnedactivity(dFF(currentcell, :)',fakerewards,timedFF,range,binsize);


% plot(x,y(:,1))
plot(x,y)
hold on
plot(x,meany,'k-','LineWidth',2)
title(['Cell ' num2str(currentcell)])
savefig(['J:\E186_plots\FiringPatternwTime\230102\Cell' num2str(currentcell)])
pause(0.8)
close
hold off
end
%%

figure;
for currentcell =6
% currentcell = 1;

thresh = 0;
colordff = dFF(:,currentcell);
colordff(colordff<thresh) = 0;
scatter(timedFF,ybinned,8,colordff,'filled')
colormap(winter)
caxis([0 1])
hold on
% scatter(timedFF(find(licks)),ybinned(find(licks)),8,'b','filled')
scatter(timedFF(find(rewards)),ybinned(find(rewards)),8,'r','filled')
title(['Cell ' num2str(currentcell)])

pause(0.8)
hold off
end


%%
figure; 
temp = corrcoef([F(3,:); Fsmooth(28,:)]');
plot(rescale(F(3,:),0,1))
hold on
plot(rescale(Fsmooth(28,:),1,2))
text(4.3,1.8,['r = ' num2str(temp(1,2))])
%%
figure;
plot(ybinnedSmooth, 'r')
hold on
plot(ybinned)