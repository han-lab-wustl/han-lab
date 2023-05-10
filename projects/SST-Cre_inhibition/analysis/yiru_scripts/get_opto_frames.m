function [vrfl] = get_opto_frames(vrfl,fmatfl)
% Zahra's function of scientist code to get opto frames
% actually a very useful function

close all
vr=load(vrfl); fm=load(fmatfl);

figure; imagesc(fm.ops.meanImg); colormap(gray); axis image
cell_pos=reshape(cell2mat(cellfun(@(x) double(x.med), fm.stat,'UniformOutput', 0)),2,size(fm.iscell,1))';
hold on
plot(cell_pos(:,2),cell_pos(:,1),'r.')
x_coords=cell_pos(:,2); y_coords=cell_pos(:,1);
%%%
opt_stim_cells=cell_pos(find(y_coords<100),:);
opt_stim_idz=find(y_coords<100);

plot(opt_stim_cells(:,2),opt_stim_cells(:,1),'wo','MarkerSize',6)
opt_tmstamp=[];
for jj=1:length(opt_stim_idz)
    %    plot(stat{1,opt_stim_idz(jj)}.xpix,stat{1,opt_stim_idz(jj)}.ypix)
    %%%%
    y=fm.Fneu(opt_stim_idz(jj),:);
    %     Fs = 32000; % Sampling Frequency.
    %     [s,f,t,pxx] = spectrogram(y,128,120,128,Fs);
    %      x=findchangepts(pow2db(pxx),'MaxNumChanges',2)
    x=findchangepts(y,'MaxNumChanges',2);
    if length(x)>1
    opt_tmstamp=[opt_tmstamp ;x];
    end
end

figure; stackedplot(fm.Fneu(opt_stim_idz,:)')
allopto_F=fm.Fneu(opt_stim_idz,:)';
x=mode(opt_tmstamp);%%%Find the common range

figure; subplot(1,2,1),stackedplot(fm.Fneu(opt_stim_idz,x(1)-200:x(2)+200)')
subplot(1,2,2),stackedplot(fm.F(opt_stim_idz,x(1)-200:x(2)+200)')

figure; subplot(1,2,1),stackedplot(fm.Fneu(opt_stim_idz,x(1)-200:x(2)+200)')
subplot(1,2,2),stackedplot(fm.F(opt_stim_idz,x(1)-200:x(2)+200)')

% figure
%  for roii = 1:size(F,1)
%  plot(rescale(F(roii,:),roii-1,roii),'LineWidth',1.5)
%  hold on
%
%  end
%  ylims=ylim; xlims=xlim;
%  patch_cords=[[x(1) 0];[x(2) 0]; [x(2) ylims(2)]; [x(1) ylims(2)]];
% patch(patch_cords(:,1),patch_cords(:,2),[0.7,0.7,0.7]); alpha(0.3)





%%
%%% split cells 50 cells each into multiple figures and plot coords in opt stim window
%%%%plot behavior
%%%sort cells by 'y' location
% close all
[val ids ]=sort(y_coords);

sort_F=fm.F(ids,:);

numcells=size(fm.F,1);
steps=1:50:numcells;
for jj=1:length(steps)-1
    find_figure(strcat('ncells_split',num2str(jj)));
    
    for roii = steps(jj):steps(jj+1)-1
        plot(rescale(sort_F(roii,:),roii-1,roii),'LineWidth',1.5)
        hold on
        
        if roii == steps(jj)
            plot(rescale(fm.licks,steps(jj)-10,steps(jj)-5),'Color',[0.7 0.7 0.7],'LineWidth',1.5)
            %
            
            plot(rescale(fm.ybinned,steps(jj)-5,steps(jj)-1),'k','LineWidth',1.5)
            plot(rescale(fm.forwardvel,steps(jj+1)+0,steps(jj+1)+5),'k','LineWidth',1.5)
            plot(rescale(fm.rewards,steps(jj)-10,steps(jj+1)+5),'g-','LineWidth',1)
            changeRewLoc(find(fm.changeRewLoc~=0))=1;
            plot(rescale(fm.changeRewLoc,steps(jj)-10,steps(jj+1)+5),'b--','LineWidth',1.5)
        end
        
    end
    
    ylims=ylim; xlims=xlim;
    patch_cords=[[x(1) ylims(1)];[x(2) ylims(1)]; [x(2) ylims(2)]; [x(1) ylims(2)]];
    patch(patch_cords(:,1),patch_cords(:,2),[0.7,0.7,0.7]); alpha(0.3)
    opto_stim=zeros(1,size(fm.F,2));
    opto_stim(x(1):x(2))=1; % Munni: opto_stim(x)=1
    vr.VR.optotrigger=opto_stim;
    save(vrfl,'vr','-append')
end
disp([vrfl ' Done!']);

end