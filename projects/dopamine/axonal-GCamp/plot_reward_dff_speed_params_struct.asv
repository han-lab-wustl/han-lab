% Zahra
% axonal-dopamine analysis
% pavlovian task
% after running batch and vrdarkrewards script (but not using any of the
% figures...)
close all;
% mouse_name = "e193";
% days = [16:20];
mouse_name = "e194";
days = [15:19];
src = "X:\dopamine_imaging"; % mouse and day folders
dst = "X:\dopamine_analysis"; % saves
planes = 3;
rnng = 100; % frames around reward to get
% add function path
addpath(fullfile(pwd, "behavior_analysis")); % navigate to han-lab main folder
for dy=days    
    fmatfl = dir(fullfile(src, mouse_name, string(dy), '**\params.mat')); % finds all params files
    for pln=1:planes
        [peridffs] = plot_perireward_per_plane(mouse_name, dy,pln,fmatfl,rnng, dst);
        bigperidff{dy,pln}=peridffs;
    end
end

% plot mean of all days per plane
for pln=1:planes
    figure;
    for dy=days
        plot(normalize(bigperidff{dy,pln},2)'); hold on
    end
    xline(median(1:rnng*2),'-.b','Reward'); %{'Conditioned', 'stimulus'}
    ylabel('norm dff')
    xlabel('frames')
    title(sprintf('%s, plane %i', mouse_name, pln))
    savefig(fullfile(dst, sprintf('%s_averagedays%03d-%03d_plane%d.fig', mouse_name, ...
        min(days), max(days), pln)))
end