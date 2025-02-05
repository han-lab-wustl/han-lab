function get_all_cells_registered(pr_dir, days_involved, template_day)
days = length(pr_dir);
org_pr_dir = pr_dir;
pr_dir(template_day)=[];
fclick = dir(fullfile(pr_dir{1}, '**', '*align.mat'));
% for reg day path
% Extract the parent directory path
[parentDir, ~, ~] = fileparts(pr_dir{1});

disp('For all days...')
for plane=1:length(fclick) % asummes every day is the same num planes

    for j = 1:days-1
        ndays{j} = j;
        fclick = dir(fullfile(pr_dir{j}, '**', sprintf('plane%i',plane-1), '*_F.mat'));
        filename_1{j} = fullfile(fclick.folder, fclick.name);
        path_1{j} = fclick.folder;
        file_1{j} = fclick.name;
    end
    % template day
    plnpy = plane-1;
    fclick = dir(fullfile(org_pr_dir{template_day}, '**', sprintf('plane%i',plnpy), '*_F.mat'));
    filename_2 = fullfile(fclick.folder, fclick.name);
    path_2 = fclick.folder; file_2 = fclick.name;
    % reg file path
    regpth = dir(fullfile(parentDir, sprintf('*pln%i_all_days.mat',plane)));
    filename_3 = fullfile(regpth.folder, regpth.name);

    for j = 1:days-1
        nday = ndays{j};
        get_registered_cells_Suite2p_multiple_days(plane,nday,days_involved,filename_1{j},filename_2, ...
            filename_3,file_1{j},path_1{j},file_2,path_2);
    end
end
end