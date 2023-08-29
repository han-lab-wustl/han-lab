xmin=400;
xmax=1000;
n=20
x=xmin+rand(1,n)*(xmax-xmin);
xmin=200
xmax=600;
n=20
y=xmin+rand(1,n)*(xmax-xmin);

%%
% slice densiites
gray = [0.7 0.7 0.7];
figure;
bar(1,mean(x),'k'); hold on;
bar(2,mean(y), 'FaceColor', gray)
% plot(1, x, 'ko');
% plot(2, y, 'ko');
xlim([0 3])
xticks([1 2])
xticklabels([{'deepCA1'}, {'supCA1'}])
ylabel('Terminal density (Norm. fluor./cm^3)')
% ylabel('ACC neuron density (Cells/cm^3)')
% fontsize(16,"points")
set(gca,'fontsize', 16) 

[h,p,~,stat]=ttest(x,y)

%%
xmin=0.1;
xmax=0.4;
n=10
x=xmin+rand(1,n)*(xmax-xmin);
xmin=0.02
xmax=0.2;
n=10
y=xmin+rand(1,n)*(xmax-xmin);

figure; 
bar(1, mean(y), 'k'); hold on;
errorbar(1,mean(y),std(y),'b')
bar(2, mean(x), 'FaceColor', gray)
errorbar(2,mean(x),std(x),'b')
set(gca,'fontsize', 16) 

xlim([0 3])
xticks([1 2])
xticklabels([{'stGTACHR2 (light on-off)'}, {'mCherry (light on-off)'}])
ylabel(sprintf('Proportion of place cells \n (w/in 10cm of reward)'))
[h,p,~,stat]=ttest(x,y)

%%
xmin=0;
xmax=0.1;
n=10;
x=xmin+rand(1,n)*(xmax-xmin);
xmin=0.2;
xmax=0.5;
n=10;
y=xmin+rand(1,n)*(xmax-xmin);

figure; 
bar(1, mean(y), 'k'); hold on;
errorbar(1,mean(y),std(y),'b')
bar(2, mean(x), 'FaceColor', gray)
errorbar(2,mean(x),std(x),'b')
set(gca,'fontsize', 16) 

xlim([0 3])
xticks([1 2])
xticklabels([{'ChrimsonR (light on-off)'}, {'mCherry (light on-off)'}])
ylabel(sprintf('Proportion of place cells \n (w/in 10cm of reward)'))
[h,p,~,stat]=ttest(x,y)