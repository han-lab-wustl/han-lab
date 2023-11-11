function plot_mean_speed(data)
% Show the mean speed changes of trials of days. One figure contains lines of eps, opto eps use dashed line
% 4x1 subplot (mean, mean(nonstop), mean(dark), mean(1/3 track)

for this_day = 1:size(data,1)
    eps = data(this_day).day_eps;
    if length([eps.epoch])<2
        continue
    end
    figure('visible','on','Renderer','painters','Position', [20 20 1000 700])
    t = tiledlayout('flow');
    xt = [];
    lg = strings(1,length([eps.epoch]));
    opto = [-1 -1];
    nexttile
    for i = 1:length([eps.epoch])
        os = eps(i).opto_stim;
        linestyle = '-';
        if sum(os)>0
            opto = find(os == 1);
            lg(length([eps.epoch])) = 'Opto Stim';
            linestyle = '--';
        end
        plot(eps(i).mean_speed, linestyle)
        hold on
        xt = [xt length(eps(i).mean_speed)];
        lg(i) = ['ep' num2str(i)];
    end
    xlim([0 max(xt)+1])
    xticks(1:max(xt))
    if sum(opto) >= 0
        yl = get(gca,'YLim');
        ylim(yl)
        patch([opto(1) opto(end) opto(end) opto(1)], [yl(1) yl(1) yl(2) yl(2)],'g','FaceAlpha',0.08,'EdgeColor','g')
    end
    legend(lg,'color','none')
    xlabel('Trial number')
    ylabel('Speed (cm/s)')
    title(['Mean speed of trials on ' char(data(this_day).date)])
    hold off

    nexttile
    for i = 1:length([eps.epoch])
        os = eps(i).opto_stim;
        linestyle = '-';
        if sum(os)>0
            opto = find(os == 1);
            linestyle = '--';
        end
        plot(eps(i).mean_speed_no_stop, linestyle)
        hold on
    end
    xlim([0 max(xt)+1])
    xticks(1:max(xt))
    if sum(opto) >= 0
        yl = get(gca,'YLim');
        ylim(yl)
        patch([opto(1) opto(end) opto(end) opto(1)], [yl(1) yl(1) yl(2) yl(2)],'g','FaceAlpha',0.08,'EdgeColor','g')
    end
    legend(lg,'color','none')
    xlabel('Trial number')
    ylabel('Speed (cm/s)')
    title(['Mean speed (nonstop) of trials on ' char(data(this_day).date)])
    hold off

    nexttile
    for i = 1:length([eps.epoch])
        os = eps(i).opto_stim;
        linestyle = '-';
        if sum(os)>0
            opto = find(os == 1);
            linestyle = '--';
        end
        plot(eps(i).mean_speed_in_dark, linestyle)
        hold on
    end
    xlim([0 max(xt)+1])
    xticks(1:max(xt))
    if sum(opto) >= 0
        yl = get(gca,'YLim');
        ylim(yl)
        patch([opto(1) opto(end) opto(end) opto(1)], [yl(1) yl(1) yl(2) yl(2)],'g','FaceAlpha',0.08,'EdgeColor','g')
    end
    legend(lg,'color','none')
    xlabel('Trial number')
    ylabel('Speed (cm/s)')
    title(['Mean speed (in dark) of trials on ' char(data(this_day).date)])
    hold off

    nexttile
    for i = 1:length([eps.epoch])
        os = eps(i).opto_stim;
        linestyle = '-';
        if sum(os)>0
            opto = find(os == 1);
            linestyle = '--';
        end
        plot(eps(i).mean_speed_one_third_track, linestyle)
        hold on
    end
    xlim([0 max(xt)+1])
    xticks(1:max(xt))
    if sum(opto) >= 0
        yl = get(gca,'YLim');
        ylim(yl)
        patch([opto(1) opto(end) opto(end) opto(1)], [yl(1) yl(1) yl(2) yl(2)],'g','FaceAlpha',0.08,'EdgeColor','g')
    end
    legend(lg,'color','none')
    xlabel('Trial number')
    ylabel('Speed (cm/s)')
    title(['Mean speed (1/3 track) of trials on ' char(data(this_day).date)])
    hold off

%     saveas(t, ['H:\E186\E186\speed\fig\mean_speed_' char(data(this_day).date) '(' char(data(this_day).day) ').fig'], 'fig')
%     saveas(t, ['H:\E186\E186\speed\png\mean_speed_' char(data(this_day).date) '(' char(data(this_day).day) ').png'], 'png')
end
disp('Done!')
% close all

end