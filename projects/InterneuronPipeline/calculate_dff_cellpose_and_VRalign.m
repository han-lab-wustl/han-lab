%Zahra wrote this script to get cells from cell pose, calc dff, and 
%align to behavior
%also used to plot cells with behavior
clear all; clear all;
Fs = 31.25;
time = 300; % ms
for i=6:7 % days
    Fmat  = dir(fullfile('F:\E136', sprintf('D%i', i), '**', 'plane*', 'Fall.mat'));
    
    for i=1:length(Fmat)
        disp(Fmat(i).folder)
        [dff,f0] = redo_dFF_from_cellpose(fullfile(Fmat(i).folder, Fmat(i).name), ...
            Fs, time);
    
    end
end
% align to behavior
clear all;
mouse_name = "E136";
days = [5:7];
src = "F:\";

for day=days
    daypth = dir(fullfile(src, mouse_name, ...
    sprintf('D%i',day), sprintf('%s*mat', mouse_name)));%, "behavior", "vr\*.mat"));
    fmatfl = dir(fullfile(src, mouse_name, sprintf('D%i',day), '**\Fall.mat')); 
    savepthfmat = VRalign(fullfile(daypth.folder, daypth.name),fmatfl, length(fmatfl));
    disp(savepthfmat)

end
%%
% plot cells with hrz behavior

%e.g.
% look at all cells at one
% subplot(3,1,1); imagesc(normalize(dFF_iscell,1)); 
% xlim([0 length(ybinned)])
% subplot(3,1,2)
% plot(forwardvel, 'b'); xlim([0 length(ybinned)])
% subplot(3,1,3); 
% plot(ybinned, 'k'); xlim([0 length(ybinned)])
% hold on; 
% plot(find(rewards==1), ybinned(rewards==1), 'go')
%%
animal = 'E135';
corrs = {};
cell_not_corr_d = {};
days = [1:4]; % compile across all days
for d=days
    Fmat  = dir(fullfile(sprintf('G:\\%s\\Day%i', animal, d), '**', 'plane*', 'Fall.mat'));
    corrs_day = {};
    for plane=1:length(Fmat)
        load(fullfile(Fmat(plane).folder, Fmat(plane).name))
        figure;
        for cell=1:size(dFF_iscell,1)            
            subplot(size(dFF_iscell,1)+2,1,cell); plot(dFF_iscell(cell,:));                         
            axis off          
            sgtitle(sprintf('%s \n Day = %i, plane = %i', animal, ...
                d, plane))
            [corr,p] = corrcoef(dFF_iscell(cell,:), ...
                smoothdata(forwardvel, 'gaussian'));
            corrs_day{cell} = corr(1,2);
            
            if corr(1,2)<0.2
                cell_not_corr{cell} = dFF_iscell(cell,:);
            end
        end        
    end
    figure;
    subplot(2,1,1)
    plot(forwardvel, 'b'); xlim([0 length(ybinned)])
    xline(find(trialnum<3),'Alpha',0.01)
    subplot(2,1,2); 
    plot(ybinned, 'k'); xlim([0 length(ybinned)])
    hold on; 
    plot(find(rewards==1), ybinned(rewards==1), 'go')
    plot(find(licks), ybinned(licks), 'r.')
    xline(find(trialnum<3),'Alpha',0.01)
    sgtitle(sprintf('Day = %i', d))

    corrs{d} = corrs_day;
    cell_not_corr_d{d} = cell_not_corr;
    figure; hist(cell2mat(corrs_day));
    title(sprintf('%s \n correlation of dff to forwardvel', ...
        Fmat(plane).folder))
end
figure; hist(cell2mat(cellfun(@(x) cell2mat(x), corrs, 'UniformOutput', false)));
title('correlation of dff to forwardvel, all days')