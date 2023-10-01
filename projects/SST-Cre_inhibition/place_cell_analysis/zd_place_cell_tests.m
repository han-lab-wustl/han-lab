clear all; clear all;
days = [59:3:75];
cc = load('Y:\sstcre_analysis\celltrack\e201_week12-15\Results\commoncells_atleastoneactivedayperweek_4weeks_week2daymap.mat');
cc = cc.cellmap2dayacrossweeks;
ddn = 1; % counter for days that align to tracked mat file
for dd=days
    putative_pcs = {};
    pth=sprintf('Y:\\sstcre_analysis\\fmats\\e201\\days\\day%i_Fall.mat',dd);    
    fprintf('\n day = % i\n', dd)
    load(pth)
    cellindt = cc(:,1);
    cellindt = cellindt(cellindt>1);
    cellindbc = [1:size(Fc3,2)];
    [~,bordercells] = remove_border_cells_all_cells(stat);    
    cellindwithoutbc = cellindbc(~bordercells);    
    cellind=intersect(cellindt,cellindwithoutbc); %only tracked cells that are not border cells
    fc3 = Fc3(:,cellind); % remove border cells from shuffle;
    eps = find(changeRewLoc);
    eps = [eps length(changeRewLoc)]; % includes end of recording as end of a epoch
    for ep=1:length(eps)-1 % find putative place cell per epoch
        fprintf("\n Epoch %i \n", ep)
        bin = 3; track_length = 270; gainf = 3/2;
        [putative_pc] = get_place_cells_per_ep(eps,ep,ybinned,fc3,changeRewLoc, ...
        bin,track_length,gainf);
        % get place cells only
        putative_pcs{ep} = putative_pc;       
    end
    save(pth, 'cc', 'putative_pcs', 'bordercells', '-append') % save border cells for further analysis
    ddn=ddn+1;
end
%%
% generate tuning curve
% bin_size = 1;
% nbins = track_length/bin_size;
% 
% for i = 1:nbins    
%     time_in_bin{i} = time_moving(ypos_mov >= (i-1)*bin_size & ypos_mov < i*bin_size);
% end 
% 
% % make bins via suyash method
% moving_cells_activity = fc3(putative_pc,time_moving);
% fc3_pc = fc3(putative_pc,:);
% for i = 1:size(moving_cells_activity,1)
%     for bin = 1:nbins
%         cell_activity(i,bin) = mean(fc3_pc(i,time_in_bin{bin})); 
%     end
% end 
% cell_activity(isnan(cell_activity)) = 0;
% figure; imagesc(normalize(cell_activity))
% 
% % sort by max value
% for c=1:length(cell_activity)
%     f = cell_activity(c,:);
%     if sum(f)>0
%         [peakval,peakbin] = max(f);
%         peak(c) = peakbin;
%     else
%         peak(c) = 1;
%     end
% end
% 
% [~,sorted_idx] = sort(peak);
% figure; imagesc(normalize(cell_activity(sorted_idx,:)))
% save(pth, 'putative_pc', '-append')
