clear all; clear all;
% days = [63:70, 72:74, 76, 81:90];
days = [25:29];
% cc = load('Y:\sstcre_analysis\celltrack\e201_week12-15\Results\commoncells_atleastoneactivedayperweek_4weeks_week2daymap.mat');
% cc = load('Y:\sstcre_analysis\celltrack\e200_week09-14\Results\commoncells_atleastoneactivedayperweek_4weeks_week2daymap.mat');
% cc = cc.cellmap2dayacrossweeks;
% ddn = 1; % counter for days that align to tracked mat file
an = 'e218';
src = 'X:\vipcre'; % folder where fall is
bin = 3; track_length = 270; gainf = 3/2; 
plns = [0]; Fs = 31.25/length(plns); % multiplane analysis not implemented
for dy=days
    putative_pcs = {};
    pth = dir(fullfile(src, an, string(dy), '**\*Fall.mat'));
    % load vars
    load(fullfile(pth.folder,pth.name), 'Fc3', 'stat', 'iscell', 'ybinned', 'changeRewLoc', ...
        'forwardvel', 'licks', 'trialnum', 'rewards')    
%     pth=dir(fullfile('Y:\\sstcre_analysis\\fmats\\e200\\days\\', sprintf('*day%03d_Fall.mat',dd)));    
    savepth = fullfile(pth.folder, pth.name);
    fprintf('\n day = % i\n', dy)
%     cellindt = cc(:,ddn);
%     cellindt = cellindt(cellindt>1); % remove dropped cells
%     cellindbc = [1:size(Fc3,2)];
    [~,bordercells] = remove_border_cells_all_cells(stat, Fc3);    
%     cellindwithoutbc = cellindbc(~bordercells);    
%     cellind=intersect(cellindt,cellindwithoutbc); %only tracked cells that are not border cells
    pc = logical(iscell(:,1));
    [~,bordercells] = remove_border_cells_all_cells(stat, Fc3);
    bordercells_pc = bordercells(pc); % mask border cells
    fc3 = Fc3(:,pc); % only iscell
    fc3_pc = fc3(:,~bordercells_pc); % remove border cells
    if ~isempty(fc3)
        eps = find(changeRewLoc);
        eps = [eps length(changeRewLoc)]; % includes end of recording as end of a epoch
        for ep=1:length(eps)-1 % find putative place cell per epoch
            fprintf("\n Epoch %i \n", ep)            
            [putative_pc] = get_place_cells_per_ep(eps,ep,ybinned,forwardvel, ...
                fc3_pc,changeRewLoc,bin,track_length,gainf,Fs);
            % get place cells only
            putative_pcs{ep} = putative_pc;       
        end
        save(savepth, 'putative_pcs', 'bordercells', '-append') % save border cells for further analysis        
    end
end
