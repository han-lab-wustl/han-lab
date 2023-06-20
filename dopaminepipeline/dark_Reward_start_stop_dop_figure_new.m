%%%% moving successful


% files=[2:3 5:8 10 12:17 21 24 30:33];%%for e156
% files=[18:31];%%for e157
% files=[40:44];%%for e158
% files=[26:30];%%for e149
% files=days_check(1:end);
% set(gcf,'renderer','Painters')
% files(1)=[];
close all
% clear all
mouse = 'E195';
find_figure('dop_days_loc_planes');clf
find_figure('fail vs success moving-triggered');clf
find_figure('fail vs success stop-triggered'); clf
find_figure('success moving-triggered activity over days');clf
find_figure('save_individual_axes'); clf
    color={[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
    setylimmanual = [0.994 1.005];
    setxlimmanual = [1 size(squeeze(roe_allsuc_mov(files,1,:)),2)];%*10/15];
    setxlimmanualsec = [-5 5];

for jj=1:3
    find_figure('dop_days_loc_planes')
    ax(jj)=subplot(6,4,1),imagesc(squeeze(roe_allsuc_mov(files,jj,:))); hold on
    plot([40 40],[0 length(files)],'Linewidth',2); colormap(ax(jj),gray);  set(gca,'xtick',[])
    xlim(setxlimmanual)
    
    text(10, min(ylim), 'Stopped', 'Horiz','left', 'Vert','bottom')
    text(62, min(ylim), 'Moving', 'Horiz','right', 'Vert','bottom')
    ylabel('Days')
    yticks(1:18:19)
    ax(jj)=subplot(6,4,(jj+1-1)*4+1),imagesc(squeeze(dop_allsuc_mov(files,jj,:))); hold on
      caxis([0.995 1.005])
    plot([40 40],[0 length(files)],'Linewidth',2); colormap(ax(jj),jet); set(gca,'xtick',[])
%     title(strcat('success_Plane',num2str(jj),'__Start Triggered'));set(gca,'xtick',[])
       xlim(setxlimmanual)
    ylabel('Days')
    yticks(1:18:19)
    xticks([])
    

    xax=frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,dop_success_perimov;
    yax=mean(squeeze(dop_allsuc_mov(files,jj,:)),1);
    if jj == 1
        minyax = min(yax);
        maxyax = max(yax);
    else
    minyax = min(min(yax),minyax);
     maxyax = max(max(yax),maxyax);
    end
    se_yax=std(squeeze(dop_allsuc_mov(files,jj,:)),1)./sqrt(size(squeeze(roe_allsuc_mov(files,jj,:)),1))
    subplot(6,4,(6-1)*4+1),hold on, 
    h10 = shadedErrorBar(xax,yax,se_yax,[],1);
    h10.patch.FaceColor = color{jj}; h10.mainLine.Color = color{jj}; h10.edge(1).Color = color{jj}; h10.edge(2).Color=color{jj}; ylim(setylimmanual); %yticks([])
    ylims = ylim;
    if jj == 4
    pls = plot([0 0],ylims,'--k','Linewidth',1);
    pls.Color(4) = 0.5;
    end
    clear xlabel; xlabel('Time(s)');
%     set(gca,'ylim',[0.99 1.01])
    if jj == 1
    plot(xax,rescale(mean(squeeze(roe_allsuc_mov(files,jj,:)),1),minyax,maxyax),'k')
    end
       xlim(setxlimmanualsec)
    find_figure('fail vs success moving-triggered');
    xax=frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,dop_success_perimov;
    yax=mean(squeeze(dop_allsuc_mov(files,jj,:)),1);
    se_yax=std(squeeze(dop_allsuc_mov(files,jj,:)),1)./sqrt(size(squeeze(roe_allsuc_mov(files,jj,:)),1))
    subplot(6,4,jj),hold on ;h10 = shadedErrorBar(xax,yax,se_yax,[],1);     h10.patch.FaceColor = color{jj}; h10.mainLine.Color = color{jj}; h10.edge(1).Color = color{jj}; h10.edge(2).Color=color{jj}; ylim(setylimmanual); yticks([])
    plot([0 0],[min(yax) max(yax)],'k','Linewidth',2)
    xlabel('Time(s)');
%     set(gca,'ylim',[0.99 1.005])
    plot(xax,rescale(mean(squeeze(roe_allsuc_mov(files,jj,:)),1),minyax,maxyax),'k') 
       xlim(setxlimmanualsec)
end


%%%%%%%%% stopping triggered successful

find_figure('dop_days_loc_planes');

for jj=1:3
    find_figure('dop_days_loc_planes')
    ax(jj)=subplot(6,4,3),imagesc(squeeze(roe_allsuc_stop(files,jj,:))); hold on
    plot([40 40],[0 length(files)],'Linewidth',2); colormap(ax(jj),gray); set(gca,'xtick',[])
    
    text(10, min(ylim), 'Moving', 'Horiz','left', 'Vert','bottom')
    text(62, min(ylim), 'Stopped', 'Horiz','right', 'Vert','bottom');
       xlim(setxlimmanual)
%     ylabel('Days')
    yticks(1:18:19)
    ax(jj)=subplot(6,4,(jj+1-1)*4+3),imagesc(squeeze(dop_allsuc_stop(files,jj,:))); hold on
     caxis([0.995 1.005])
    plot([40 40],[0 length(files)],'Linewidth',2); colormap(ax(jj),jet); set(gca,'xtick',[])
%     title(strcat('success_Plane',num2str(jj),'__Stop Triggered'));
       xlim(setxlimmanual)
%     ylabel('Days')
    yticks(1:18:19)
    xax=frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,dop_success_peristop;
    yax=mean(squeeze(dop_allsuc_stop(files,jj,:)),1);
     if jj == 1
        minyax = min(yax);
        maxyax = max(yax);
    else
    minyax = min(min(yax),minyax);
     maxyax = max(max(yax),maxyax);
    end
    se_yax=std(squeeze(dop_allsuc_stop(files,jj,:)),1)./sqrt(size(squeeze(roe_allsuc_stop(files,jj,:)),1))
    subplot(6,4,(6-1)*4+3),hold on, ;h10 = shadedErrorBar(xax,yax,se_yax,[],1);     h10.patch.FaceColor = color{jj}; h10.mainLine.Color = color{jj}; h10.edge(1).Color = color{jj}; h10.edge(2).Color=color{jj}; ylim(setylimmanual); yticks([])

    %     hp=shadedErrorBar(xax,yax,se_yax,'Color',color{jj},1)
    %     hp=plot(yax,'Color',color{jj})
    %     legend(hp,strcat('plane',num2str(jj)));hold on
    xlabel('Time(s)'); 
       xlim(setxlimmanualsec)
%     set(gca,'ylim',[0.99 1.01])
    if jj == 1
    plot(xax,rescale(mean(squeeze(roe_allsuc_stop(files,jj,:)),1),minyax,maxyax),'k')
    end
    ylims = ylim;
    if jj == 4
         ylims = ylim;
    pls = plot([0 0],ylims,'--k','Linewidth',1);
    ylim(ylims)
    pls.Color(4) = 0.5;
    end
    
    find_figure('fail vs success stop-triggered');
    xax=frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,dop_success_peristop;
    yax=mean(squeeze(dop_allsuc_stop(files,jj,:)),1);
    se_yax=std(squeeze(dop_allsuc_stop(files,jj,:)),1)./sqrt(size(squeeze(roe_allsuc_stop(files,jj,:)),1))
    subplot(6,4,jj),hold on, ;h10 = shadedErrorBar(xax,yax,se_yax,[],1);     h10.patch.FaceColor = color{jj}; h10.mainLine.Color = color{jj}; h10.edge(1).Color = color{jj}; h10.edge(2).Color=color{jj}; ylim(setylimmanual); yticks([])
    ylims = ylim;
    pls = plot([0 0],ylims,'--k','Linewidth',1);
    pls.Color(4) = 0.5;
    %     hp=shadedErrorBar(xax,yax,se_yax,'Color',color{jj},1)
    %     hp=plot(yax,'Color',color{jj})
    %     legend(hp,strcat('plane',num2str(jj)));hold on
    xlabel('Time(s)'); 
%     set(gca,'ylim',[0.99 1.01])
    plot(xax,rescale(mean(squeeze(roe_allsuc_stop(files,jj,:)),1),0.99,1),'k')
    title(strcat('success__allPlane__Stop Triggered'));
       xlim(setxlimmanualsec)
end


 %%%%perireward
      setylimmanual = [0.984 1.01];
    for jj=1:3
        find_figure('perirew_dop_days_loc_planes')
        ax(jj)=subplot(6,4,1),imagesc(squeeze(roe_allsuc_perireward(files,jj,:))); hold on
        plot([40 40],[0 length(files)],'Linewidth',2); colormap(ax(jj),gray);  set(gca,'xtick',[])
        xlim(setxlimmanual)
        
        text(10, min(ylim), 'Stopped', 'Horiz','left', 'Vert','bottom')
        text(62, min(ylim), 'Moving', 'Horiz','right', 'Vert','bottom')
        ylabel('Days')
        yticks(1:18:19)
        ax(jj)=subplot(6,4,(jj+1-1)*4+1),imagesc(squeeze(dop_allsuc_perireward(files,jj,:))); hold on
        caxis([0.995 1.005])
        plot([40 40],[0 length(files)],'Linewidth',2); colormap(ax(jj),jet); set(gca,'xtick',[])
        %     title(strcat('success_Plane',num2str(jj),'__Start Triggered'));set(gca,'xtick',[])
        xlim(setxlimmanual)
        ylabel('Days')
        yticks(1:18:19)
        xticks([])
        
        
        xax=frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,dop_success_perimov;
        yax=mean(squeeze(dop_allsuc_perireward(files,jj,:)),1);
        if jj == 1
            minyax = min(yax);
            maxyax = max(yax);
        else
            minyax = min(min(yax),minyax);
            maxyax = max(max(yax),maxyax);
        end
        se_yax=std(squeeze(dop_allsuc_perireward(files,jj,:)),1)./sqrt(size(squeeze(roe_allsuc_perireward(files,jj,:)),1))
        subplot(6,4,(6-1)*4+1),hold on,
        h10 = shadedErrorBar(xax,yax,se_yax,[],1);
        h10.patch.FaceColor = color{jj}; h10.mainLine.Color = color{jj}; h10.edge(1).Color = color{jj}; h10.edge(2).Color=color{jj}; ylim(setylimmanual); %yticks([])
        ylims = ylim;
        if jj == 4
            pls = plot([0 0],ylims,'--k','Linewidth',1);
            pls.Color(4) = 0.5;
        end
        clear xlabel; xlabel('Time(s)');
        %     set(gca,'ylim',[0.99 1.01])
        if jj == 1
            plot(xax,rescale(mean(squeeze(roe_allsuc_perireward(files,jj,:)),1),minyax,maxyax),'k')
        end
        xlim(setxlimmanualsec)
        find_figure('perirew_fail vs success moving-triggered');
        xax=frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,dop_success_perimov;
        yax=mean(squeeze(dop_allsuc_perireward(files,jj,:)),1);
        se_yax=std(squeeze(dop_allsuc_perireward(files,jj,:)),1)./sqrt(size(squeeze(roe_allsuc_perireward(files,jj,:)),1))
        subplot(6,4,jj),hold on ;h10 = shadedErrorBar(xax,yax,se_yax,[],1);     h10.patch.FaceColor = color{jj}; h10.mainLine.Color = color{jj}; h10.edge(1).Color = color{jj}; h10.edge(2).Color=color{jj}; ylim(setylimmanual); yticks([])
        plot([0 0],[min(yax) max(yax)],'k','Linewidth',2)
        xlabel('Time(s)');
        %     set(gca,'ylim',[0.99 1.005])
        plot(xax,rescale(mean(squeeze(roe_allsuc_perireward(files,jj,:)),1),minyax,maxyax),'k')
        xlim(setxlimmanualsec)
    end
    
    
    
    