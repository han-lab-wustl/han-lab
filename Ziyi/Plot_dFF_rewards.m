clear;  load('E:\Ziyi\Data\240326_ZH\240326_ZH_000_000\suite2p\plane3\reg_tif\file0000_XC_plane_4_roibyclick_F.mat')

frameLength = 60;

% Plotting

for n = 1:size(dFF,2) % n is number of rois

    dFF_signal = dFF(:,n); % dFF for that roi
    

    % Find each reward location 
    csLoc = cellfun(@(x) x(1),consecutive_stretch(find(rewards==1)),'Un',1);
    usLoc = cellfun(@(x) x(1),consecutive_stretch(find(solenoid2==1)),'Un',1);

    dFF_period = zeros(frameLength*2+1,size(csLoc,2));

    for i = 1:length(csLoc)
        index = csLoc(i);

        % Define start and end indices, considering the boundaries
        startIdx = max(index - frameLength, 1);
        endIdx = min(index + frameLength, length(dFF_signal));
        dFF_period(:,i) = dFF_signal(startIdx:endIdx);

    end

    subplot(round(size(dFF,2)/2),2, n);
    %plot(mean(dFF_period')); hold on;
    %smoothdata(F(10,:),'gaussian',5)
    plot(smoothdata(mean(dFF_period'),'gaussian',5)); hold on;
    plot([frameLength frameLength], ylim, ':k')
    title(sprintf('Average dFF ROI%d', n));
end