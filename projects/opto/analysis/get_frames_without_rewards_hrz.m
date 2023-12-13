function [nonrew_frames, exclude_frames_all] = get_frames_without_rewards_hrz(rewards)

% 10 s after rew to look for non rewarded licks
exclude_frames = find(rewards==1); % exclude these frames and those around them
exclude_frames_all = {};
for i=1:length(exclude_frames)
    if length(rewards)>exclude_frames(i)+300
        exclude_frames_all{i} = exclude_frames(i):exclude_frames(i)+300; % ~ 10 s
    else
        exclude_frames_all{i} = exclude_frames(i):length(rewards); % ~ 10 s
    end
end
exclude_frames_all = cell2mat(exclude_frames_all);
frames = 1:length(rewards);
nonrew_frames = ismember(frames, exclude_frames_all);
nonrew_frames = frames(nonrew_frames);
end