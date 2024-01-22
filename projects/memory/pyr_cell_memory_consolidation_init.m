clear all; 
% zahra's quick analysis
mouse_name = "e217";
days = [8,9,10];
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

%     daypth = dir(fullfile(src, mouse_name, sprintf('%i',day), "behavior", "vr\*.mat"));
% %     sprintf('%i',day), sprintf('%s*mat', mouse_name)));%,     
%     savepthfmat = VRalign(fullfile(daypth.folder, daypth.name),fmatfl, length(fmatfl));
%     disp(savepthfmat)
end
%%
figure; 
F_ = F(logical(iscell(:,1)),:);
plot(ybinned); hold on
plot(find(licks), ybinned(licks), 'r.')
plot(find(rewards), ybinned(logical(rewards)), 'b.')
yyaxis right 
plot(dFF(:,randi([1 size(dFF,2)]),:),'g');
% plot(F_(randi([1 size(F_,1)]),:),'g');