function recreate_iscell_and_make_all_struct(Settings)

for this_day = 1:size(Settings.paths,1)

    clearvars -except this_day Settings
%     should_be_analyzed = 1;

    file =fullfile(Settings.paths(this_day).folder,Settings.paths(this_day).name);
    directory = file;
    info = split(directory,'\');
    mouse_cd = string(info{Settings.level_mouse_name});
    day_cd = num2str(string(info{Settings.level_day})); % num2str for zahra's folder structure
    day=str2num(day_cd(2:end));

   l = load(file);
%    skewdcells = find(skewness(l.F,1,2)<2); %looks at skewness of cells, <2 --> interneurons; if omitted, gets all cells
%    l.iscell(skewdcells,1) = 0;
   disp ([file ' ... loaded'])
    %don't remove iscells==0, unnecessary for this analysis
    %filter by only common cells
    %e.g. first day
%     cells = ismember([1:size(l.iscell)],Settings.commoncells.cc(:,day))'; 
%     cells = ones(1,size(l.iscell));
    remove_iscell = [];
    this_real_cell = 0;
    for this_cell = 1: size(l.F,1)
        if l.iscell(this_cell,1)==1
            this_real_cell = this_real_cell+1;
            f_this_cell = l.F(this_cell,:);
            f0_this_cell = l.Fneu(this_cell,:);
            if sum(f_this_cell)==0

                plot(f_this_cell)
                hold on
                plot(f0_this_cell)
                hold off
                remove_iscell(this_real_cell) = 1;
                drawnow;
            else
                remove_iscell(this_real_cell) = 0;
            end
        end
    end

    try
        all=l.all; % ZD added
    end
    if ~exist('all','var')
        disp('problem with all variable ')
        disp('..recreating all structure.. ')
        cells = l.iscell(:,1)== 1 ;
        cd(Settings.paths(this_day).folder)
        all = create_all_structure(l.F,l.Fneu,l.spks,Settings.Fs,cells);
        save(file , 'all','-append') 
        disp('done recreating all !') 
    end
    disp ([file ' ... done!'])
end

end