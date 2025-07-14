function [putative_pcs] = get_place_cells_all_ep_w_bordercells(stat, Fc3, iscell, ...
    changeRewLoc, ybinned,forwardvel,bin,track_length,gainf,Fs,pth)
%     cellindwithoutbc = cellindbc(~bordercells);    
%     cellind=intersect(cellindt,cellindwithoutbc); %only tracked cells that are not border cells
pc = logical(iscell(:,1));
[~,bordercells] = remove_border_cells_all_cells(stat, Fc3);
% bordercells_pc = bordercells(pc); % mask border cells
fc3_pc = Fc3(:,pc); % only iscell
% fc3_pc = fc3(:,~bordercells_pc); % remove border cells
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
    save(pth, 'putative_pcs', 'bordercells', '-append') % save border cells for further analysis        
end    
end