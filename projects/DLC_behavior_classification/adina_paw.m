%filter paw data
badPawTop_y = find(PawTop_likelihood<0.9);
PawTop_y_fil = PawTop_y;
PawTop_y_fil(badPawTop_y)=0;
badPawMiddle_y = find(PawMiddle_likelihood<0.9);
PawMiddle_y_fil = PawMiddle_y;
PawMiddle_y_fil(badPawMiddle_y)=0;
badPawBottom_y = find(PawBottom_likelihood<0.9);
PawBottom_y_fil = PawBottom_y;
PawBottom_y_fil(badPawBottom_y)=0;
%average paw y values
paw_mean_y = ((PawTop_y_fil+PawMiddle_y_fil+PawBottom_y_fil)/3);
figure;
plot(paw_mean_y)

paw_b=diff(paw_mean_y>0);
hold on
plot(paw_b)
starts=find(paw_b==1);
stops=find(paw_b==-1);
bouts = stops-starts;
plot(ybinned)
plot(rescale(rewards,0,10))