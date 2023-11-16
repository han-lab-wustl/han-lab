% dlc curation
% get a list of .mat files that have pressed keys, etc.
% remove those from analysis
% 331 = single reward ascii code
clear all
src = 'Y:\DLC\dlc_mixedmodel2';
matfls = dir(fullfile(src, "*time*.mat"));
for fl=1:length(matfls)
    vr = load(fullfile(matfls(fl).folder, matfls(fl).name));
    keys = vr.VR.pressedKeys(vr.VR.pressedKeys>0);
    if length(keys)>1
        if sum(keys==55)>0 || sum(keys==56)>0 || sum(keys==57)>0 || sum(keys==331)>0
            % if rew zone was changed or lick contingency switched off before task                        
%             disp(keys)
            delete(fullfile(matfls(fl).folder, matfls(fl).name))
            disp(matfls(fl).name) 
        end
    end
end