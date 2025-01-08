% Constants
load('E:\Ziyi\results\E227_E169_comparison\superficial_planes_Bulk_all_mice.mat')
load('E:\Ziyi\results\E227_E169_comparison\superficial_planes_E227_first_three_day_interneuron.mat')
frame_rate = 7.8; % Frame rate in Hz
num_seconds = 5; % Number of seconds to display from -5 to +5 around the reward
pre_win = 5;
post_win = 5;
pre_win_frames = round(pre_win * frame_rate);
post_win_frames = round(post_win * frame_rate);

planecolors={[110 0 95]/256,[253 141 60]/256};

% Assuming the middle point (0 seconds) is at frame 40 of the extracted -39 to +39 frames
mid_point = 40; % This is frame 0 on your original scale
%frame_ticks = mid_point + (-num_seconds:num_seconds) * frame_rate; % Frame positions for each second from -5 to +5
frame_ticks = [1 40 79];
time_ticks = -5:5:5; % Time labels from -5 to +5

g = 10;
figure('Position', [100, 100, 400, 300])
% Plotting the first variable: superPlanes_E169_A
mean_data_A = mean(superPlanes_Bulk, 2);
norm_mean_data_A = mean_data_A./mean(mean_data_A(1:pre_win_frames));
yax_A = g*(norm_mean_data_A-1)+1;
se_yax_A = g*std(superPlanes_Bulk,[],2)./sqrt(size(superPlanes_Bulk,2))';
xax = -pre_win_frames:post_win_frames;
h10_A = shadedErrorBar(xax', yax_A, se_yax_A, [], 1);

% Formatting for the first plot
% xt=[-3*ones(1,length(reg_name))];
% yt=[0.992:0.001:1];

xticks(frame_ticks - mid_point) % Shift ticks to align with the plotted data range
xticklabels(time_ticks) % Label ticks from -5 to +5 seconds

if sum(isnan(se_yax_A)) ~= length(se_yax_A)
    h10_A.patch.FaceColor = planecolors{1}; 
    h10_A.mainLine.Color = planecolors{1}; 
    h10_A.edge(1).Color = planecolors{1};
    h10_A.edge(2).Color = planecolors{1};
end
hold on

% Plotting the second variable: superPlanes_E169_B
mean_data_B = mean(superPlanes_E227, 2);
norm_mean_data_B = mean_data_B./mean(mean_data_B(1:pre_win_frames));
yax_B = norm_mean_data_B;
se_yax_B = std(superPlanes_E227,[],2)./sqrt(size(superPlanes_E227,2))';
h10_B = shadedErrorBar(xax', yax_B, se_yax_B, [], 1);
% 
% Formatting for the second plot
if sum(isnan(se_yax_B)) ~= length(se_yax_B)
    h10_B.patch.FaceColor = planecolors{2}; 
    h10_B.mainLine.Color = planecolors{2}; 
    h10_B.edge(1).Color = planecolors{2};
    h10_B.edge(2).Color = planecolors{2};
end

% Common plot formatting
ylims = ylim;
pls = plot([0 0],ylims,'--k','Linewidth',1);
ylim(ylims)
pls.Color(4) = 0.5;

% Adding the legend
%legend({'Bulk', 'Sparse'}, 'Location', 'Best');

% Adding the legend with custom labels and colors
legend([h10_A.mainLine, h10_B.mainLine], {'bulk', 'sparse'}, 'Location', 'Best', "FontSize",14);
box off
hold off
axis off
% set(gcf, 'color', 'none');    
% set(gca, 'color', 'none');

