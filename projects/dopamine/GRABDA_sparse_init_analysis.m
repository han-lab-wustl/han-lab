% zahra's quick analysis
mouse_name = "e228";
days = [10];
src = "Y:\hrz_consolidation";
%% calc dff and append to behavior
for day=days
    fmatfl = dir(fullfile(src, mouse_name, sprintf('%i',day), '**\Fall.mat')); 
    load(fullfile(fmatfl.folder, fmatfl.name)); % assumes single plane
    % make dff of iscells
    F_ = F(logical(iscell(:,1)),:);
    Fneu_ = Fneu(logical(iscell(:,1)),:);
    [dFF,Fc3] = Calc_Fc3_Reverse_Subtraction(F_,Fneu_,31.25);
    save(fullfile(fmatfl.folder, fmatfl.name), 'dFF', 'Fc3', '-append')

    daypth = dir(fullfile(src, mouse_name, sprintf('%i',day), "behavior", "vr\*.mat"));
%     sprintf('%i',day), sprintf('%s*mat', mouse_name)));%,     
    savepthfmat = VRalign(fullfile(daypth.folder, daypth.name),fmatfl, length(fmatfl));
    disp(savepthfmat)
end
%% peri reward analysis
load(fullfile(fmatfl.folder, fmatfl.name))
[binnedvel,~,~] = perirewardbinnedvelocity(forwardvel,rewards,timedFF,10,0.1);
[binnedlicks,~,~] = perirewardbinnedvelocity(licks,rewards,timedFF,10,0.1);
[binnedPerireward,allbins,rewdFF,normmeanrewdFF] = perirewardbinnedactivity(dFF,rewards,timedFF,10,0.1);
% plot dff with beh
figure
subplot(2,1,1)
plot(binnedvel, 'b')
ylabel('velocity')
subplot(2,1,2)
plot(binnedlicks, 'r')
ylabel('lick rate')
for j=1:size(rewdFF,2)
    figure;
    for i=1:size(rewdFF,3) % per roi
    plot(rewdFF(:,j,i)); hold on
    end
    plot(binnedPerireward(j,:), 'k')
end
ylabel('dFF')
