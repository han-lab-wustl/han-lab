
setxlimmanualsec = [-5 5];
setylimmanual = [0.983 1.05];
currROI_labels = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
 xt=[-3*ones(1,6)];
    yt=[1.005:0.003:1.02];
exday = 2;
color={[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
currtitle = 'E168';
  roerescale = [0.985 0.995];
    maxspeedlim = 25; %cm/s
figure; 

            for jj = 1:4
            xax=frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes;
            yax=nanmean(roi_dop_alldays_planes_perireward{exday,jj}',1);
            if jj == 1
                minyax = min(yax);
                maxyax = max(yax);
            else
                minyax = min(min(yax),minyax);
                maxyax = max(max(yax),maxyax);
            end
            se_yax=nanstd(roi_dop_alldays_planes_perireward{exday,jj}',[],1)./sqrt(size(roi_dop_alldays_planes_perireward{exday,jj}',1));
            hold on, ;h10 = shadedErrorBar(xax,yax,se_yax,[],1);
            if sum(isnan(se_yax))~=length(se_yax)
                h10.patch.FaceColor = color{jj}; h10.mainLine.Color = color{jj}; h10.edge(1).Color = color{jj};
                h10.edge(2).Color=color{jj};
                text(xt(jj),yt(jj),currROI_labels{jj},'Color',color{jj})
                h10.patch.FaceAlpha = 0.07;
                h10.mainLine.LineWidth = 1.5;
                h10.edge(1).Color(4) = 0.07;
                h10.edge(2).Color(4) = 0.07;
%                 h10.edge(1).LineWidth =
            end
           
            title(currtitle)
            ylim(setylimmanual);

            xlim(setxlimmanualsec)
%             allmouse_time{2}{currmouse,jj} = xax;
            if jj == 1
                
                if size(roe_success_peristop,2)==79
                    xax=frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames;
                else
                    xax=frame_time*(-pre_win_frames)*numplanes:frame_time:frame_time*post_win_frames*numplanes;
                end
                plot(xax,nanmean(roe_alldays_planes_perireward{exday,jj}',1)/maxspeedlim*diff(roerescale)+roerescale(1),'k')
            end
            ylims = ylim;
            if jj == size(roi_dop_alldays_planes_perireward,2)
                ylims = ylim;
                pls = plot([0 0],ylims,'--k','Linewidth',1);
                ylim(ylims)
                pls.Color(4) = 0.5;
            end
            end
            plot([-3 -3],[1.03 1.04],'k-','LineWidth',2)