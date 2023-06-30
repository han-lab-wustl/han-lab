<<<<<<< HEAD

% clearvars -except mouseP mouse SST_ints data
% load('C:\Users\Han Lab\MA F Files\Saved Variable mats\PVstructWithGLM.mat')


%note changed a lot from suyash's version. PLEASE do not run unless you
%read the entire thing and make sure that all variables are correct.

%This requires a mouseP structure from neuron structurors. Please make sure
%you have done this first.

mouseP_num = 1;
othervariables = ({'CPP','masks','mimg','name'}); %variables that relate to the expirment but do not run throughout it such as name meanimage masks or CPP
% other varialbes are manually set for now, but can be automatically set using some indicator such as the sizes don't match or you have a way of identifying them all. PLEEEASE update this every time you run to make sure
experiment = 'R1';
%% No remap Days
D_idx = [2 3 4 5];
D_Fs = [10.4 ];
 
% Behavior on respective days - R1
% ypos
nDay = size(D_idx,2);
nMouse = length(mouseP_num);
%VIP_ints = struct;
for mouses = 1:nMouse
    for day_num = 1:nDay
        cell = D_idx(mouses,day_num);
        mousePIdx = mouseP_num(mouses);
        
        
        expirementvariables = fieldnames(mouseP(mousePIdx)); %variables that have the same length and run throughout the trials such as ypos or velocity or Falls
        for rem = 1:size(othervariables,2)
            removeIndx = cell2mat(cellfun(@(x) strcmp(x,othervariables(rem)),expirementvariables,'un',0));
            expirementvariables(find(removeIndx)) = [];
        end
 
        
        cppIdx = 1;
        for cp = 1:length(mouseP(mousePIdx).CPP{cell})
            cpp = mouseP(mousePIdx).CPP{cell}(cp);
            
            RewEpochstarts = find(mouseP(mousePIdx).rewardLocation{cell}(:,cppIdx));
          
            nRewEpoch = length(RewEpochstarts);
            [ans1,ans2] = find(mouseP(mousePIdx).rewardLocation{cell}(:,:));
            epochmatrix = reshape(ans1,nRewEpoch,length(ans1)/nRewEpoch);
            for rewepoch = 1:nRewEpoch %divide into reward epochs and probe epochs
                if rewepoch == nRewEpoch %assigning all experiment values for the last reward location using end
                        for var = 1:size(expirementvariables,1) 
                            if strcmp(expirementvariables{var},'F0')
                                VIP_ints(mouses).(experiment).day{day_num}.RewEp{rewepoch}.(expirementvariables{var}){cp}.raw =...
                                    mouseP(mousePIdx).(expirementvariables{var}){cell}.neuro{1}(RewEpochstarts(rewepoch):end,cppIdx:cppIdx+cpp-1);
                                VIP_ints(mouses).(experiment).day{day_num}.RewEp{rewepoch}.(expirementvariables{var}){cp}.raw =...
                                    mouseP(mousePIdx).(expirementvariables{var}){cell}.neuro{1}(RewEpochstarts(rewepoch):end,cppIdx:cppIdx+cpp-1);
                            else
                                if iscell(mouseP(mousePIdx).(expirementvariables{var}){cell})
                                    VIP_ints(mouses).(experiment).day{day_num}.RewEp{rewepoch}.(expirementvariables{var}){cp} =...
                                        mouseP(mousePIdx).(expirementvariables{var}){cell}{1}(RewEpochstarts(rewepoch):end,cppIdx:cppIdx+cpp-1);
                                else
                                    VIP_ints(mouses).(experiment).day{day_num}.RewEp{rewepoch}.(expirementvariables{var}){cp} =...
                                        mouseP(mousePIdx).(expirementvariables{var}){cell}(RewEpochstarts(rewepoch):end,cppIdx:cppIdx+cpp-1);
                                end
                            end
%                             assignin('base',['VIP_ints(mouses).(experiment).day{day_num}.RewEp{rewepoch}.' expirementvariables{var} '{cell}{cp}'],...
%                                 eval(['mouseP(mousePIdx).' expirementvariables{var} '{cell}(RewEpochstarts(rewepoch):end,cppIdx:cppIdx+cpp-1)']));
                        end
                    
                else        %assigning all experiment values for other reward locations
                        for var = 1:size(expirementvariables,1)
                            if strcmp(expirementvariables{var},'F0')
                                VIP_ints(mouses).(experiment).day{day_num}.RewEp{rewepoch}.(expirementvariables{var}){cp}.raw =...
                                    mouseP(mousePIdx).(expirementvariables{var}){cell}.neuro{1}(RewEpochstarts(rewepoch):RewEpochstarts(rewepoch+1)-1,cppIdx:cppIdx+cpp-1);
                                VIP_ints(mouses).(experiment).day{day_num}.RewEp{rewepoch}.(expirementvariables{var}){cp}.raw =...
                                    mouseP(mousePIdx).(expirementvariables{var}){cell}.neuro{1}(RewEpochstarts(rewepoch):RewEpochstarts(rewepoch+1)-1,cppIdx:cppIdx+cpp-1);
                            else
                                if iscell(mouseP(mousePIdx).(expirementvariables{var}){cell})
                                    VIP_ints(mouses).(experiment).day{day_num}.RewEp{rewepoch}.(expirementvariables{var}){cp} =...
                                        mouseP(mousePIdx).(expirementvariables{var}){cell}{1}(RewEpochstarts(rewepoch):RewEpochstarts(rewepoch+1)-1,cppIdx:cppIdx+cpp-1);
                                else
                                    VIP_ints(mouses).(experiment).day{day_num}.RewEp{rewepoch}.(expirementvariables{var}){cp} =...
                                        mouseP(mousePIdx).(expirementvariables{var}){cell}(RewEpochstarts(rewepoch):RewEpochstarts(rewepoch+1)-1,cppIdx:cppIdx+cpp-1);
                                end
                            end
                        end
            
                end
                for varother = 1:size(othervariables,2)
                    VIP_ints(mouses).(experiment).day{day_num}.RewEp{rewepoch}.(othervariables{varother}) =...
                        mouseP(mousePIdx).(othervariables{varother}){cell};
                    
                end
            end
            
            %keep an all format for the entire session
            for var = 1:size(expirementvariables,1)
                if strcmp(expirementvariables{var},'F0')
                    VIP_ints(mouses).(experiment).day{day_num}.all.(expirementvariables{var}).raw(:,cppIdx:cppIdx+cpp-1) =...
                        mouseP(mousePIdx).(expirementvariables{var}){cell}.raw{1}(:,cppIdx:cppIdx+cpp-1);
                                        VIP_ints(mouses).(experiment).day{day_num}.all.(expirementvariables{var}).neuro(:,cppIdx:cppIdx+cpp-1) =...
                        mouseP(mousePIdx).(expirementvariables{var}){cell}.neuro{1}(:,cppIdx:cppIdx+cpp-1);
                else
                    if iscell(mouseP(mousePIdx).(expirementvariables{var}){cell})
                        VIP_ints(mouses).(experiment).day{day_num}.all.(expirementvariables{var}){cp} =...
                            mouseP(mousePIdx).(expirementvariables{var}){cell}{1}(:,cppIdx:cppIdx+cpp-1);
                    else
                        VIP_ints(mouses).(experiment).day{day_num}.all.(expirementvariables{var}){cp} =...
                            mouseP(mousePIdx).(expirementvariables{var}){cell}(:,cppIdx:cppIdx+cpp-1);
                    end
                end
            end
            for varother = 1:size(othervariables,2)
                VIP_ints(mouses).(experiment).day{day_num}.all.(othervariables{varother}) =...
                    mouseP(mousePIdx).(othervariables{varother}){cell};
                
            end
            cppIdx = cppIdx + cpp;
        end
        VIP_ints(mouses).(experiment).day{day_num}.Fs = D_Fs(mouses,day_num);
    end
end

=======

% clearvars -except mouseP mouse SST_ints data
% load('C:\Users\Han Lab\MA F Files\Saved Variable mats\PVstructWithGLM.mat')


%note changed a lot from suyash's version. PLEASE do not run unless you
%read the entire thing and make sure that all variables are correct.

%This requires a mouseP structure from neuron structurors. Please make sure
%you have done this first.

mouseP_num = 1;
othervariables = ({'CPP','masks','mimg','name'}); %variables that relate to the expirment but do not run throughout it such as name meanimage masks or CPP
% other varialbes are manually set for now, but can be automatically set using some indicator such as the sizes don't match or you have a way of identifying them all. PLEEEASE update this every time you run to make sure
experiment = 'R1';
%% No remap Days
D_idx = [2 3 4 5];
D_Fs = [10.4 ];
 
% Behavior on respective days - R1
% ypos
nDay = size(D_idx,2);
nMouse = length(mouseP_num);
%VIP_ints = struct;
for mouses = 1:nMouse
    for day_num = 1:nDay
        cell = D_idx(mouses,day_num);
        mousePIdx = mouseP_num(mouses);
        
        
        expirementvariables = fieldnames(mouseP(mousePIdx)); %variables that have the same length and run throughout the trials such as ypos or velocity or Falls
        for rem = 1:size(othervariables,2)
            removeIndx = cell2mat(cellfun(@(x) strcmp(x,othervariables(rem)),expirementvariables,'un',0));
            expirementvariables(find(removeIndx)) = [];
        end
 
        
        cppIdx = 1;
        for cp = 1:length(mouseP(mousePIdx).CPP{cell})
            cpp = mouseP(mousePIdx).CPP{cell}(cp);
            
            RewEpochstarts = find(mouseP(mousePIdx).rewardLocation{cell}(:,cppIdx));
          
            nRewEpoch = length(RewEpochstarts);
            [ans1,ans2] = find(mouseP(mousePIdx).rewardLocation{cell}(:,:));
            epochmatrix = reshape(ans1,nRewEpoch,length(ans1)/nRewEpoch);
            for rewepoch = 1:nRewEpoch %divide into reward epochs and probe epochs
                if rewepoch == nRewEpoch %assigning all experiment values for the last reward location using end
                        for var = 1:size(expirementvariables,1) 
                            if strcmp(expirementvariables{var},'F0')
                                VIP_ints(mouses).(experiment).day{day_num}.RewEp{rewepoch}.(expirementvariables{var}){cp}.raw =...
                                    mouseP(mousePIdx).(expirementvariables{var}){cell}.neuro{1}(RewEpochstarts(rewepoch):end,cppIdx:cppIdx+cpp-1);
                                VIP_ints(mouses).(experiment).day{day_num}.RewEp{rewepoch}.(expirementvariables{var}){cp}.raw =...
                                    mouseP(mousePIdx).(expirementvariables{var}){cell}.neuro{1}(RewEpochstarts(rewepoch):end,cppIdx:cppIdx+cpp-1);
                            else
                                if iscell(mouseP(mousePIdx).(expirementvariables{var}){cell})
                                    VIP_ints(mouses).(experiment).day{day_num}.RewEp{rewepoch}.(expirementvariables{var}){cp} =...
                                        mouseP(mousePIdx).(expirementvariables{var}){cell}{1}(RewEpochstarts(rewepoch):end,cppIdx:cppIdx+cpp-1);
                                else
                                    VIP_ints(mouses).(experiment).day{day_num}.RewEp{rewepoch}.(expirementvariables{var}){cp} =...
                                        mouseP(mousePIdx).(expirementvariables{var}){cell}(RewEpochstarts(rewepoch):end,cppIdx:cppIdx+cpp-1);
                                end
                            end
%                             assignin('base',['VIP_ints(mouses).(experiment).day{day_num}.RewEp{rewepoch}.' expirementvariables{var} '{cell}{cp}'],...
%                                 eval(['mouseP(mousePIdx).' expirementvariables{var} '{cell}(RewEpochstarts(rewepoch):end,cppIdx:cppIdx+cpp-1)']));
                        end
                    
                else        %assigning all experiment values for other reward locations
                        for var = 1:size(expirementvariables,1)
                            if strcmp(expirementvariables{var},'F0')
                                VIP_ints(mouses).(experiment).day{day_num}.RewEp{rewepoch}.(expirementvariables{var}){cp}.raw =...
                                    mouseP(mousePIdx).(expirementvariables{var}){cell}.neuro{1}(RewEpochstarts(rewepoch):RewEpochstarts(rewepoch+1)-1,cppIdx:cppIdx+cpp-1);
                                VIP_ints(mouses).(experiment).day{day_num}.RewEp{rewepoch}.(expirementvariables{var}){cp}.raw =...
                                    mouseP(mousePIdx).(expirementvariables{var}){cell}.neuro{1}(RewEpochstarts(rewepoch):RewEpochstarts(rewepoch+1)-1,cppIdx:cppIdx+cpp-1);
                            else
                                if iscell(mouseP(mousePIdx).(expirementvariables{var}){cell})
                                    VIP_ints(mouses).(experiment).day{day_num}.RewEp{rewepoch}.(expirementvariables{var}){cp} =...
                                        mouseP(mousePIdx).(expirementvariables{var}){cell}{1}(RewEpochstarts(rewepoch):RewEpochstarts(rewepoch+1)-1,cppIdx:cppIdx+cpp-1);
                                else
                                    VIP_ints(mouses).(experiment).day{day_num}.RewEp{rewepoch}.(expirementvariables{var}){cp} =...
                                        mouseP(mousePIdx).(expirementvariables{var}){cell}(RewEpochstarts(rewepoch):RewEpochstarts(rewepoch+1)-1,cppIdx:cppIdx+cpp-1);
                                end
                            end
                        end
            
                end
                for varother = 1:size(othervariables,2)
                    VIP_ints(mouses).(experiment).day{day_num}.RewEp{rewepoch}.(othervariables{varother}) =...
                        mouseP(mousePIdx).(othervariables{varother}){cell};
                    
                end
            end
            
            %keep an all format for the entire session
            for var = 1:size(expirementvariables,1)
                if strcmp(expirementvariables{var},'F0')
                    VIP_ints(mouses).(experiment).day{day_num}.all.(expirementvariables{var}).raw(:,cppIdx:cppIdx+cpp-1) =...
                        mouseP(mousePIdx).(expirementvariables{var}){cell}.raw{1}(:,cppIdx:cppIdx+cpp-1);
                                        VIP_ints(mouses).(experiment).day{day_num}.all.(expirementvariables{var}).neuro(:,cppIdx:cppIdx+cpp-1) =...
                        mouseP(mousePIdx).(expirementvariables{var}){cell}.neuro{1}(:,cppIdx:cppIdx+cpp-1);
                else
                    if iscell(mouseP(mousePIdx).(expirementvariables{var}){cell})
                        VIP_ints(mouses).(experiment).day{day_num}.all.(expirementvariables{var}){cp} =...
                            mouseP(mousePIdx).(expirementvariables{var}){cell}{1}(:,cppIdx:cppIdx+cpp-1);
                    else
                        VIP_ints(mouses).(experiment).day{day_num}.all.(expirementvariables{var}){cp} =...
                            mouseP(mousePIdx).(expirementvariables{var}){cell}(:,cppIdx:cppIdx+cpp-1);
                    end
                end
            end
            for varother = 1:size(othervariables,2)
                VIP_ints(mouses).(experiment).day{day_num}.all.(othervariables{varother}) =...
                    mouseP(mousePIdx).(othervariables{varother}){cell};
                
            end
            cppIdx = cppIdx + cpp;
        end
        VIP_ints(mouses).(experiment).day{day_num}.Fs = D_Fs(mouses,day_num);
    end
end

>>>>>>> 754f532e47d152334ffae033cf3e5763ab9bf2c0
