% find vr files with specific tasks
% pooling data for behavior analysis
clear all; close all
src = '\\storage1.ris.wustl.edu\ebhan\Active\all_vr_data';
dst = '\\storage1.ris.wustl.edu\ebhan\Active\dzahra\dlc';
fls = dir(fullfile(src, "e*.mat"));
an_rr = {}; nmr = 1; an_hrz = {}; nmhrz = 1;
for fl=1:length(fls)  
    vr = load(fullfile(fls(fl).folder,fls(fl).name));    
    if isfield(vr.VR,'settings') && isfield(vr.VR, 'pressedKeys')
        keys = vr.VR.pressedKeys(vr.VR.pressedKeys>0);
        exp = vr.VR.settings.name;
        eps = find(vr.VR.changeRewLoc>0);
        eps = [eps length(vr.VR.changeRewLoc)];
        if contains(exp,"Random")
            an_rr{nmr} = fullfile(fls(fl).folder,fls(fl).name);
            nmr=nmr+1;        
        elseif contains(exp,"HRZ")            
            if length(eps)>3 % at least 2 ep                 
            if length(keys)>1 % no manual intervention                
                if sum(keys==55)==0 && sum(keys==56)==0 && sum(keys==57)==0 && sum(keys==331)==0
                disp(fullfile(fls(fl).folder,fls(fl).name))
                an_hrz{nmhrz} = fullfile(fls(fl).folder,fls(fl).name);
                nmhrz=nmhrz+1;
                ypos = vr.VR.ypos*(1/vr.VR.scalingFACTOR);                
                velocity = vr.VR.ROE(2:end)*-0.013./diff(vr.VR.time);                                
                try
                fig = figure; 
                plot(ypos); hold on; 
                plot(vr.VR.changeRewLoc*(1/vr.VR.scalingFACTOR), 'b', 'LineWidth',3)
                plot(find(vr.VR.lick),ypos(find(vr.VR.lick)), ...
                    'k.', 'MarkerSize',5) 
                plot(find(vr.VR.reward),ypos(find(vr.VR.reward)), ...
                    'ko', 'MarkerSize',5) 
                ylabel("track length (cm)")
                xlabel("frames")
                title(vr.VR.name_date_vr) 
                saveas(gcf, fullfile(dst, strcat(vr.VR.name_date_vr,'.jpg')))
                close(fig)
                catch
                end
                end
            end
            end
        end
    end
    clear vr
end

rr_t = cell2table(an_rr');
hrz_t = cell2table(an_hrz');
% save out mat files
writetable(rr_t, fullfile(src, 'random_reward.csv'))
writetable(hrz_t, fullfile(src, 'hrz.csv'))
