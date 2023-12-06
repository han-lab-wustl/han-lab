clear all; clear all; 
animal = 'e218';
srcdir = 'X:\vipcre';
days = [38];
ep=2;
for d=days
    fallm = dir(fullfile(srcdir,animal,sprintf('%i',d), '**', 'plane*','Fall.mat'));
    for pl=1:length(fallm)
        load(fullfile(fallm(pl).folder,fallm(pl).name))
        Fall = load(fullfile(fallm(pl).folder,fallm(pl).name));
        if ep==1
            eprng = eps(1):eps(2);
        elseif ep==2
            eprng = eps(2):eps(3);
        elseif ep==3        
            eprng = eps(3):eps(4);        
        end

%     eprng_comp_m = ones(1,length(changeRewLoc));%eps(1):eps(2);
%     eprng_comp_m(eprng) = 0; % remove opto ep but keep the rest
%     rng = [1:length(changeRewLoc)];
        eprng_comp = eprng;%rng(logical(eprng_comp_m));
    
        % get xy axis
        stat_iscell = stat(logical(Fall.iscell(:,1)));
        if ~(size(stat_iscell,2)==size(F,1)) % check if same size as all.dff (sometimes cells are not removed) 
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


        dff = all.dff(~logical(topbordercells),:); dffs{d} = dff;
        eps = find(changeRewLoc);
        eps = [eps length(changeRewLoc)];
        
        data_ = mean(dff,1,'omitnan');%smoothdata(mean(dff,1,'omitnan'), 'gaussian',20);
        figure; subplot(3,1,1);
        plot(data_,'k'); hold on;
        for mm = 1:length(eps)-1 %the rectangle indicating the reward location, overlaps the probe trials referring to the previous reward location
            rectangle('position',[eps(mm) min(data_) length(find(trialnum(eps(mm):eps(mm+1)-1)<3)) max(data_)-min(data_)], ... % just picked max for visualization
                'EdgeColor',[0 0 0 0],'FaceColor',[0 .5 .5 0.3])
        end
        xlim([0 length(data_)])
        subplot(3,1,2)
        plot(forwardvel, 'b'); xlim([0 length(ybinned)]); hold on;
        for mm = 1:length(eps)-1 %the rectangle indicating the reward location, overlaps the probe trials referring to the previous reward location
            rectangle('position',[eps(mm) min(data_) length(find(trialnum(eps(mm):eps(mm+1)-1)<3)) max(data_)-min(data_)], ... % just picked max for visualization
                'EdgeColor',[0 0 0 0],'FaceColor',[0 .5 .5 0.3])
        end
        
        subplot(3,1,3); 
        plot(ybinned, 'k'); xlim([0 length(ybinned)])
        hold on; 
        plot(find(rewards==1), ybinned(rewards==1), 'go')
        plot(find(licks), ybinned(licks), 'r.')
        for mm = 1:length(eps)-1 %the rectangle indicating the reward location, overlaps the probe trials referring to the previous reward location
            rectangle('position',[eps(mm) min(data_) length(find(trialnum(eps(mm):eps(mm+1)-1)<3)) max(data_)-min(data_)], ... % just picked max for visualization
                'EdgeColor',[0 0 0 0],'FaceColor',[0 .5 .5 0.3])
        end
        sgtitle(sprintf('animal = %s, day = %i', animal,d))
    end
end