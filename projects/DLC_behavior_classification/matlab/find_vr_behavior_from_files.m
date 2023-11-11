% find vr files with specific tasks
% pooling data for behavior analysis
clear all; close all
src = 'Y:\DLC\VR_data\dlc\new_vids';
dst = 'Y:\DLC\VR_data';
fls = dir(fullfile(src, "*.mat"));
an_rr = {}; nmr = 1; an_hrz = {}; nmhrz = 1;
for fl=1:length(fls)
    vr = load(fullfile(fls(fl).folder,fls(fl).name));
    if isfield(vr.VR,'settings')
        exp = vr.VR.settings.name;
        if contains(exp,"Random")
            an_rr{nmr} = vr.VR.name_date_vr;
            nmr=nmr+1;        
        elseif contains(exp,"HRZ")
            an_hrz{nmhrz} = vr.VR.name_date_vr;
            nmhrz=nmhrz+1;
            ypos = vr.VR.ypos*(1/vr.VR.scalingFACTOR);
            
            velocity = vr.VR.ROE(2:end)*-0.013./diff(vr.VR.time);
            eps = find(vr.VR.changeRewLoc>0);
            eps = [eps length(vr.VR.changeRewLoc)];
            if length(eps)>3
                fig = figure; 
                plot(ypos, 'k','Marker', '.', 'MarkerSize',2); hold on; 
                plot(vr.VR.changeRewLoc*(1/vr.VR.scalingFACTOR), 'b', 'LineWidth',3)
                plot(find(vr.VR.lick),ypos(find(vr.VR.lick)), ...
                    'r.', 'MarkerSize',5) 
                ylabel("track length (cm)")
                xlabel("frames")
                title(vr.VR.name_date_vr) 
                savefig(fullfile(dst, strcat(vr.VR.name_date_vr,'.fig')))
                close(fig)
            end
        end
    end
end

rr_t = cell2table(an_rr');
hrz_t = cell2table(an_hrz');

writetable(rr_t, fullfile(src, 'random_reward.csv'))
writetable(hrz_t, fullfile(src, 'hrz.csv'))
