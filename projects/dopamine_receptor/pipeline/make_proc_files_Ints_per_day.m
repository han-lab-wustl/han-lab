
function make_proc_files_Ints_per_day(pr_dir)

for dy=1:length(pr_dir) % per day
% zd added
    fclick = dir(fullfile(pr_dir{dy}, '**', '*roibyclick_F.mat'));
    fall = dir(fullfile(pr_dir{dy}, '**', 'plane*\*Fall.mat'));
    for plane = 1:length(fall)
        make_proc_files_from_F_file_Ints(plane,dy, fclick(plane), fall(plane))
    end
end
end
