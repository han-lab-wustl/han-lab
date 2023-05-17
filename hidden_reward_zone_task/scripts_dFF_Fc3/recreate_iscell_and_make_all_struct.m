function recreate_iscell_and_make_all_struct(Settings)

figure('Renderer', 'painters', 'Position', [50 50 1000 300])
for this_day = 1:size(Settings.paths,1)

    clearvars -except this_day Settings
%     should_be_analyzed = 1;

    file = fullfile(Settings.paths(this_day).folder,Settings.paths(this_day).name);
%     directory = file;
%     info = split(directory,'/');
%     mouse_cd = string(info{Settings.level_mouse_name});
%     day_cd = string(info{Settings.level_day});

   l = load(file);
%    skewdcells = find(skewness(l.F(:,:)')<2);%EB GM W
%    l.iscell(skewdcells,1) = 0;
   disp ([file ' ... loaded'])

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
    if ~exist('all','var') || size(all.dff,1) ~= size(remove_iscell,2)
        disp('problem with all variable ')
        disp('..recreating all structure.. ')
        
       cells = l.iscell(:,1)== 1 ;

       cd(Settings.paths(this_day).folder)
       
       
       all = create_all_structure(l.F,l.Fneu,l.spks,Settings.Fs,cells);
       
       % ZD removed, filters out a LOT of cells in e200
%        skews=[];
%        skews = skewness(all.dff,1,2);
%        removed.Fc3 = all.Fc3(skews<2,:);
%        removed.Spks = all.Spks(skews<2,:);
%        removed.dff = all.dff(skews<2,:);
%        all.Fc3(skews<2,:) = [];
%        all.Spks(skews<2,:)=[];
%        all.dff(skews<2,:) = [];
        save( "Fall.mat" , 'all','-append') 
%         save("Fall.mat",'removed','-append')
        disp('done recreating all !')
    end
        save( file , 'remove_iscell','-append')
    
    disp ([file ' ... done!'])
end
end