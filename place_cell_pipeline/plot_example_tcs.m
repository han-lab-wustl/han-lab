i=1 ; %comparison

set(gca,'FontName','Arial')  % Set it to arail
comparison = comparisons(i,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%fig 1%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig = figure('Renderer', 'painters','Position', [30 30 500 400]);
subplot(1,3,1)
plt = tuning_curves{comparison(1)};
[~,sorted_idx] = sort(coms{comparison(1)}); % sorts first tuning curve rel to another
imagesc(normalize(plt(sorted_idx,:),2));
% plot rectangle of rew loc
% everything divided by 3 (bins of 3cm)
rectangle('position',[ceil(rewlocs(comparison(1))/bin_size)-ceil((rew_zone/bin_size)/2) 0 ...
    rew_zone/bin_size size(plt,1)], ...
    'EdgeColor',[0 0 0 0],'FaceColor',[1 1 1 0.5])
xticks([0:bin_size*3:ceil(track_length/bin_size)])
xticklabels([0:bin_size*3*bin_size:track_length])
title(sprintf('Epoch %i', comparison(1)))
xlabel('Position (cm)')
ylabel('Place cells')
hold on;
subplot(1,3,2)
plt = tuning_curves{comparison(2)};
imagesc(normalize(plt(sorted_idx,:),2));
% plot rectangle of rew loc
% everything divided by 3 (bins of 3cm)
rectangle('position',[ceil(rewlocs(comparison(2))/bin_size)-ceil((rew_zone/bin_size)/2) 0 ...
    rew_zone/bin_size size(plt,1)], ...
    'EdgeColor',[0 0 0 0],'FaceColor',[1 1 1 0.5])
xticks([0:bin_size*3:ceil(track_length/bin_size)])
xticklabels([0:bin_size*3*bin_size:track_length])
title(sprintf('Epoch %i', comparison(2)))
sgtitle(sprintf(['animal %s, day %i \n' ...
    'ep%i vs ep%i: ranksum = %d'], an, dy, comparison(1), comparison(2),...
    p))
subplot(1,3,3)
plt = tuning_curves{comparison(2)};
[~,sorted_idx] = sort(coms{comparison(2)}); % sorts first tuning curve rel to another
imagesc(normalize(plt(sorted_idx,:),2));
% plot rectangle of rew loc
% everything divided by 3 (bins of 3cm)
rectangle('position',[ceil(rewlocs(comparison(2))/bin_size)-ceil((rew_zone/bin_size)/2) 0 ...
    rew_zone/bin_size size(plt,1)], ...
    'EdgeColor',[0 0 0 0],'FaceColor',[1 1 1 0.5])
xticks([0:bin_size*3:ceil(track_length/bin_size)])
xticklabels([0:bin_size*3*bin_size:track_length])
cls = 1:length(sorted_idx);
yticklabels(cls(sorted_idx))
title(sprintf('Epoch %i resorted', comparison(2)))
sgtitle('VIP Inhibition')

set(gca,'FontName','Arial')  % Set it to arail