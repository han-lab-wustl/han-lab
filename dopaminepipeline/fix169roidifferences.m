clear all
load('D:\workspaces_darkreward\169_dark_reward_AllDays_workspace.mat')
%%
% make 169 all days only 4 rois that match to the same regions
daykey = repmat(1:4,26,1);
daykey(7,:) = [1 2 4 5];
daykey(10,:) = [1 2 4 5];
daykey(18,:) = [1 4 7 8];
daykey(19,:) = [1 4 7 8];

allvariables = who;

for allv = 1:length(allvariables)
    if size(eval(allvariables{allv}),2) == 8
        currvar = eval(allvariables{allv});
        clear newvar
        if length(size(currvar))==2
            for d = 1:size(currvar,1)
            newvar(d,:) = currvar(d,daykey(d,:));
            end
        elseif length(size(currvar))==3
            
            for d = 1:size(currvar,1)
            newvar(d,:,:) = currvar(d,daykey(d,:),:);
            end
        end
        eval([allvariables{allv} ' = newvar;']);
    end
    
    
end
%% 169
for p = 1:4 % fix motion correction day
    roi_dop_alldays_planes_periCS{21,p}(64,34) = nanmean([roi_dop_alldays_planes_periCS{21,p}(63,34) roi_dop_alldays_planes_periCS{21,p}(65,34)]);
    roi_dop_allsuc_perirewardCS(21,p,:) = nanmean(roi_dop_alldays_planes_periCS{21,p},2);
    
    roi_dop_alldays_planes_perireward{21,p}(60,34) = nanmean([roi_dop_alldays_planes_perireward{21,p}(59,34) roi_dop_alldays_planes_perireward{21,p}(61,34)]);
    roi_dop_allsuc_perireward(21,p,:) = nanmean(roi_dop_alldays_planes_perireward{21,p},2);
    
    roi_dop_alldays_planes_perireward_0{21,p}(60,34) = nanmean([roi_dop_alldays_planes_perireward_0{21,p}(59,34) roi_dop_alldays_planes_perireward{21,p}(61,34)]);
    roi_dop_allsuc_perireward(21,p,:) = nanmean(roi_dop_alldays_planes_perireward{21,p},2);
    
    roi_dop_alldays_planes_periUS{21,p}(60,34) = nanmean([roi_dop_alldays_planes_periUS{21,p}(59,34) roi_dop_alldays_planes_periUS{21,p}(61,34)]);
    roi_dop_allsuc_perirewardUS(21,p,:) = nanmean(roi_dop_alldays_planes_periUS{21,p},2);
    
    roi_dop_alldays_planes_success_mov{21,p}(54,18) = nanmean([ roi_dop_alldays_planes_success_mov{21,p}(54,17) roi_dop_alldays_planes_success_mov{21,p}(54,19)]);
    roi_dop_allsuc_mov(21,p,:) = nanmean(roi_dop_alldays_planes_success_mov{21,p});
    
    roi_dop_alldays_planes_success_mov_reward{21,p}(37,18) = nanmean([ roi_dop_alldays_planes_success_mov_reward{21,p}(37,17) roi_dop_alldays_planes_success_mov_reward{21,p}(37,19)]);
    roi_dop_allsuc_mov_reward(21,p,:) = nanmean(roi_dop_alldays_planes_success_mov_reward{21,p});
    
    roi_dop_alldays_planes_success_stop{21,p}(54,63) = nanmean([ roi_dop_alldays_planes_success_stop{21,p}(54,62) roi_dop_alldays_planes_success_stop{21,p}(54,64)]);
    roi_dop_allsuc_stop(21,p,:) = nanmean(roi_dop_alldays_planes_success_stop{21,p});
    
    roi_dop_alldays_planes_success_stop_reward{21,p}(37,63) = nanmean([ roi_dop_alldays_planes_success_stop_reward{21,p}(37,62) roi_dop_alldays_planes_success_stop_reward{21,p}(37,64)]);
    roi_dop_allsuc_stop_reward(21,p,:) = nanmean(roi_dop_alldays_planes_success_stop_reward{21,p});   
end


save('D:\workspaces_darkreward\169_dark_reward_AllDays_CutPlanes_workspace2.mat')



%%
% make 156 all days only 4 rois that match to the same regions
daykey = repmat([1 3 5 6],14,1);
daykey(13,:) = [1 3 2 5];

allvariables = who;

for allv = 1:length(allvariables)
    if size(eval(allvariables{allv}),2) == 6
        currvar = eval(allvariables{allv});
        clear newvar
        if length(size(currvar))==2
            for d = 1:size(currvar,1)
            newvar(d,:) = currvar(d,daykey(d,:));
            end
        elseif length(size(currvar))==3
            
            for d = 1:size(currvar,1)
            newvar(d,:,:) = currvar(d,daykey(d,:),:);
            end
        end
        eval([allvariables{allv} ' = newvar;']);
    end
    
    
end

save('D:\workspaces_darkreward\156_dark_reward_AllDays_CutPlanes_workspace2.mat')


%%
% make 171 all days only 4 rois that match to the same regions
daykey = repmat([1  2 3 4],10,1);
daykey(6,:) = [1 2 3 6];

allvariables = who;

for allv = 1:length(allvariables)
    if size(eval(allvariables{allv}),2) == 7
        currvar = eval(allvariables{allv});
        clear newvar
        if length(size(currvar))==2
            for d = 1:size(currvar,1)
            newvar(d,:) = currvar(d,daykey(d,:));
            end
        elseif length(size(currvar))==3
            
            for d = 1:size(currvar,1)
            newvar(d,:,:) = currvar(d,daykey(d,:),:);
            end
        end
        eval([allvariables{allv} ' = newvar;']);
    end
    
    
end

save('D:\workspaces_darkreward\171_dark_reward_AllDays_CutPlanes_workspace2.mat')


%%
% make 167 all days only 4 rois that match to the same regions
daykey = repmat([1  2 3 4],26,1);
daykey(7,:) = [1 2 3 5];
daykey(10,:) = [1 2 5 6];
daykey(18:20,:) = repmat([1 2 4 6],3,1);

allvariables = who;

for allv = 1:length(allvariables)
    if size(eval(allvariables{allv}),2) == 7
        currvar = eval(allvariables{allv});
        clear newvar
        if length(size(currvar))==2
            for d = 1:size(currvar,1)
            newvar(d,:) = currvar(d,daykey(d,:));
            end
        elseif length(size(currvar))==3
            
            for d = 1:size(currvar,1)
            newvar(d,:,:) = currvar(d,daykey(d,:),:);
            end
        end
        eval([allvariables{allv} ' = newvar;']);
    end
    
    
end

save('D:\workspaces_darkreward\167_dark_reward_AllDays_CutPlanes_workspace2.mat')
