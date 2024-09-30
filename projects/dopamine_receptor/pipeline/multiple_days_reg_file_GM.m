% This forms 1 big matrix with registered cells numbers across all days

function multiple_days_reg_file_GM(pr_dir, template_day)

ndays = length(pr_dir);
pr_dir(template_day)=[];
fclick = dir(fullfile(pr_dir{1}, '**', '*align.mat'));

for nplane=1:length(fclick) % asummes every day is the same num planes    
    for i = 1:ndays-1 % assumes first day is template day
        fclick = dir(fullfile(pr_dir{i}, '**', sprintf('plane%i',nplane-1), '*align.mat'));
        reg_file(i) = load(fullfile(fclick.folder, fclick.name));
    end

    template_intersections = reg_file(1).regi.rois.iscell_idcs(:,2);
    if ~isempty(template_intersections > 0)

        for i = 1:ndays-1
            temp = intersect(template_intersections,reg_file(i).regi.rois.iscell_idcs(:,2));
            clear template_intersections
            template_intersections = temp;
            clear temp
        end

        all_days_cell = zeros(length(template_intersections),ndays);

        all_days_cell(:,ndays) = template_intersections;
        for i = 1:ndays-1
            day_cells = reg_file(i).regi.rois.iscell_idcs(:,1);
            temp_cells = reg_file(i).regi.rois.iscell_idcs(:,2);
            for j = 1:length(template_intersections)
                idx = find(temp_cells == template_intersections(j));
                all_days_cell(j,i) = day_cells(idx);
            end

            clear day_cells temp_cells idx
        end

        iscell = all_days_cell;

        uisave('iscell');
    end
end

