
function recreate_iscell_and_make_all_struct(Settings)

figure('Renderer', 'painters', 'Position', [50 50 1000 300])
for this_day = 1:size(Settings.paths,1)

    clearvars -except this_day Settings
    should_be_analyzed = 1;

    file = fullfile(Settings.paths(this_day).folder,Settings.paths(this_day).name);
    directory = file;
    info = split(directory,'/');
    mouse_cd = string(info{6});
    day_cd = string(info{7});

    load(file)
    disp ([file ' ... loaded'])

    remove_iscell = [];
    this_real_cell = 0;
    for this_cell = 1: size(F,1)
        if iscell(this_cell,1)==1
            this_real_cell = this_real_cell+1;
            f_this_cell = F(this_cell,:);
            f0_this_cell = Fneu(this_cell,:);
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
    if ~exist('all','var') || size(all.dff,1) ~= size(remove_iscell,2)
        disp('problem with remove iscell variable! ')
        disp('..recreating all structure.. ')
        
       cells = iscell(:,1)== 1 ;

       cd(Settings.paths(this_day).folder)
       all = create_all_structure(F,Fneu,spks,Settings.Fs,cells);
        save( "Fall.mat" , 'all','-append') 
        disp('done recreating Fall!')
    end
        save( file , 'remove_iscell','-append')
    
    disp ([file ' ... done!'])
end
end
