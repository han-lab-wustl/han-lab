% find vr files with specific tasks
% pooling data for behavior analysis
clear all
src = 'Y:\DLC\VR_data\dlc';
fls = dir(fullfile(src, "E*"));
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
        end
    end
end

rr_t = cell2table(an_rr');
hrz_t = cell2table(an_hrz');

writetable(rr_t, fullfile(src, 'random_reward.csv'))
writetable(hrz_t, fullfile(src, 'hrz.csv'))
