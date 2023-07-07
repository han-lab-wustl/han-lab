function [dffs,diff_opto,spatial_info] = collect_neural_data_opto(days,srcdir,animal,ep, ...
    plot_dff)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
for d=1:length(days)
    fmatfl = dir(fullfile(srcdir, animal, string(days(d)), "**\Fall.mat")); 
    load(fullfile(fmatfl.folder,fmatfl.name));
    Fall = load(fullfile(fmatfl.folder,fmatfl.name)); % to avoid iscell conflitcts
    
    eps = find(changeRewLoc);
    % ep 2
    if ep==2
        eprng = eps(2):eps(3);
    elseif ep==3
        try
            eprng = eps(3):eps(4);
        catch
            eprng = eps(3):length(changeRewLoc);
        end
    end

%     eprng_comp_m = ones(1,length(changeRewLoc));%eps(1):eps(2);
%     eprng_comp_m(eprng) = 0; % remove opto ep but keep the rest
%     rng = [1:length(changeRewLoc)];
    eprng_comp = eps(2):eps(3);%rng(logical(eprng_comp_m));

    % get xy axis
    stat_iscell = stat(logical(Fall.iscell(:,1)));
    if ~(size(stat_iscell,2)==size(all.dff,1)) % check if same size as all.dff (sometimes cells are not removed) 
        if exist('cell2remove', 'var') % check if cell2remove var exists
            stat_cell2remove = stat_iscell(~logical(cell2remove)&(~logical(remove_iscell)));
        else
            stat_cell2remove = stat_iscell((~logical(remove_iscell)));
        end
    else
        stat_cell2remove = stat_iscell;
    end
    Ypix = cellfun(@(x) x.ypix, stat_cell2remove, 'UniformOutput', false);
    topbordercells = zeros(1,length(Ypix)); % bool of cells at the top border
    for yy=1:length(Ypix) % idea is to remove these cells
        if sum(Ypix{yy}<100)>0
            topbordercells(yy)=1;
        end
    end
    % visualize
%     stat_topbordercells = stat_cell2remove(logical(topbordercells));
%     figure;
%     imagesc(ops.meanImg)
%     colormap('gray')
%     hold on;
%     for cell=1:length(stat_topbordercells)%length(commoncells)        
%         plot(stat_topbordercells{cell}.xpix, stat_topbordercells{cell}.ypix);         
%     end
    % only get cells > y pix of 100

    dff = all.dff(~logical(topbordercells),:); dffs{d} = dff;
    % #1 find spatial info and add to Fall
    % 10 cm bins
    bin = 10; % cm t
    track_length = 270;
    spatial_info{d} = get_spatial_info_all_cells(dff',ybinned,31.25, ...
        ceil(track_length/bin),track_length);
    save(fullfile(fmatfl.folder,fmatfl.name),'spatial_info','-append')
    neural_data = dff(:,eprng);
    mask = (trialnum(eprng)>=3) & (trialnum(eprng)<8);
    opto = neural_data(:,mask);
    mask = trialnum(eprng_comp)>=8; %
%     comparison mask for first 5 trials or rest of ep
    opto_comp = dff(:, eprng_comp);
    opto_comp = opto_comp(:,mask);
    % take mean across time
    opto_mean = mean(opto, 2);
    opto_comp_mean = mean(opto_comp, 2);
    opto_means{d} = opto_mean;
    opto_comp_means{d} = opto_comp_mean;
    % vars for plotting
    if plot_dff==1
        data_ = smoothdata(mean(dff,1,'omitnan'), 'gaussian',20);
        figure;plot(data_,'k'); hold on;
        for mm = 1:length(eps)-1 %the rectangle indicating the reward location, overlaps the probe trials referring to the previous reward location
            rectangle('position',[eps(mm) min(data_) length(find(trialnum(eps(mm):eps(mm+1)-1)<3)) quantile(data_,0.8)], ...
                'EdgeColor',[0 0 0 0],'FaceColor',[0 .5 .5 0.3])
        end
        optoep = eprng;
        rectangle('position',[min(optoep(trialnum(optoep)>=3)) min(data_) length(optoep((trialnum(optoep)>=3) & (trialnum(optoep)<8))) quantile(data_,0.8)], ...
                'EdgeColor',[0 0 0 0],'FaceColor',[1 0 0 0.3])        
        title(sprintf('animal = %s, day = %i, ep = %i', animal,days(d),ep))
    end
end

diff_opto = diff([cell2mat(cellfun(@(x) x', opto_means, 'UniformOutput', false)); ...
    cell2mat(cellfun(@(x) x', opto_comp_means, 'UniformOutput', false))]);
end