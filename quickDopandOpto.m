clear all
close all
pr_dir=uipickfiles; 
days_check=1:length(pr_dir);
planecolors={[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
timewindow = 10; %s peri window
 for days=days_check
     dir_s2p = struct2cell(dir([pr_dir{days} '\**\suite2p']));
    planefolders = dir_s2p(:,~cellfun(@isempty,regexp(dir_s2p(1,:),'plane')));
     days
for allplanes=1:size(planefolders,2) %1:4
        clearvars -except mouse_id pr_dir days_check days dir_s2p planefolders allplanes planecolors timewindow
       
        pr_dir2=strcat(planefolders{2,allplanes},'\plane',num2str(allplanes-1),'\reg_tif\')
 
        %%% grab 1st file/initial 3000 frames
        cd(pr_dir2)
        
        load('params')
        dFF = params.roibasemean3{1};
        dFF_base_mean=mean(dFF);
                dFF = dFF/dFF_base_mean;
        find_figure(['Raw Data Day ' num2str(days)])
        plot(timedFF,(dFF-min(dFF))/range(dFF)+allplanes-1,'Color',planecolors{allplanes})
        hold on
        xlims = xlim;
        plot([0.9*xlims(2) 0.9*xlims(2)],[allplanes-0.1/range(dFF) allplanes],'k-','LineWidth',1.5)
        abfstims = bwlabel(stims>0.5);
            for dw = 1:max(abfstims)-1
                if find(abfstims==dw+1,1)-find(abfstims==dw,1,'last')<0.5*1000
                    abfstims(find(abfstims==dw,1):find(abfstims==dw+1,1)) = dw+1;
                end
                
            end
         abfstims = abfstims>0.5;
            abfrect =  consecutive_stretch(find(abfstims));
        if allplanes == size(planefolders,2)
            plot(utimedFF,rescale(forwardvelALL,-1,0),'k-')
            
            ylims = ylim;
            for r = 1:length(abfrect)
                rectangle('Position',[abfrect{r}(1)/1000 ylims(1) length(abfrect{r})/1000 ylims(2)-ylims(1)],'FaceColor',[0 0.5 0.5 0.3],'EdgeColor',[00 0 0 0])
            end
            
        end
        
        find_figure(['PeriOpto Start Day ' num2str(days)])
        subplot(2,2,allplanes)
        hold on
        xs = cellfun(@(x) x(1)/1000 ,abfrect,'UniformOutput',1);
        for x = 1:length(xs)
            currx = find(timedFF>=(xs(x)-timewindow)&timedFF<(xs(x)+timewindow));
            yyaxis left
            plot(timedFF(currx)-xs(x),dFF(currx),'-','Color',planecolors{allplanes})
            ylim([0.85 1.2])
            hold on
            yyaxis right
            currspeedx = find(utimedFF>=(xs(x)-timewindow)&utimedFF<(xs(x)+timewindow));
            plot(utimedFF(currspeedx)-xs(x),forwardvelALL(currspeedx),'k-')
            ylim([-5 200])
            yticks([0:25:75])
            
        end
        title(['Plane ' num2str(allplanes)])
        
end
 end