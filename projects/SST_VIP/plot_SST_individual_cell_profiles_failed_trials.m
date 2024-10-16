% zahra adaptation of gm's code
% sst individual cell profiles
clear all; close all
an = 'e136';
dy = 5;
load(fullfile("Y:\analysis\fmats", sprintf("%s", an), sprintf("%s_day%03d_plane0_Fall.mat", an, dy)));
pln0 = load(fullfile("Y:\analysis\fmats", sprintf("%s", an), sprintf("%s_day%03d_plane0_Fall.mat", an, dy)));
pln1 = load(fullfile("Y:\analysis\fmats", sprintf("%s", an), sprintf("%s_day%03d_plane1_Fall.mat", an, dy)));
pln2 = load(fullfile("Y:\analysis\fmats", sprintf("%s", an), sprintf("%s_day%03d_plane2_Fall.mat", an, dy)));
plns = {};
plns{1} = pln0; plns{2} = pln1; plns{3} = pln2;
planes = 3;
bin_size = 9; %cm, lower causes missing vals
eps = find(changeRewLoc>0);
eps = [eps length(changeRewLoc)];
track_length = 180; %cm; TODO: import from VR instead
nbins = track_length/bin_size;
rewlocs = changeRewLoc(changeRewLoc>0);
rewsize = 10; %cm
grayColor = [.7 .7 .7]; purple = [0.4940, 0.1840, 0.5560];
savedst = 'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\figure_data\sst';
%%
% params to export to ppt
pptx    = exportToPPTX('', ...
    'Dimensions',[12 6], ...
    'Title','cell profiles', ...
    'Author','Zahra', ...
    'Subject','Automatically generated PPTX file', ...
    'Comments','This file has been automatically generated by exportToPPTX');

for ep = 1:length(eps)-1 % for each epoch
    eprng = eps(ep):eps(ep+1);
    ypos = ybinned(eprng);
    trialNum = trialnum(eprng);
    rew = rewards(eprng);
    timedff = timedFF(eprng);
    vel = forwardvel(eprng);
    [success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialNum,rew);
 
    rewloc = rewlocs(ep);    
    for pln=1:planes % for each plane
        dff = plns{pln}.dFF_iscell(:,eprng);
        success_dff_per_trial = zeros(length(ftr),nbins, size(dff,1));
        trind = 1;
        if ~isempty(ftr)
        fig = figure('Renderer', 'painters', 'WindowState', 'maximized');
        slideId = pptx.addSlide();
        fprintf('Added slide %d\n',slideId);

        for tr=ftr % for each successful trial
            trind=trind+1;
            trrng = (trialNum==tr);
            vel_ = vel(trrng);
            % bin by ypos
            for i = 1:nbins
                time_in_bin{i} = find(ypos(trrng) >= (i-1)*bin_size & ...
                    ypos(trrng) < i*bin_size);
            end
            dff_bin = zeros(nbins, size(dff,1));
            vel_bin = zeros(1,nbins);
            dff_ = dff(:,trrng);
            for i = 1:size(dff,1)
                for bin = 1:nbins
                    dff_bin(bin,i) = mean(dff_(i,time_in_bin{bin}), 'omitnan');
                    if i==1 % only needs to be binned once
                        vel_bin(bin) = mean(vel_(time_in_bin{bin}),'omitnan');
                    end
                end
            end
            % 0 out nans like suyash?
            vel_bin(isnan(vel_bin)) = 0;
            dff_bin(isnan(dff_bin)) = 0;
            for cll=1:size(dff_bin,2) % plot each cell
                subplot(ceil(sqrt(size(dff,1))),ceil(sqrt(size(dff,1))),cll)
                plot(normalize(dff_bin(:,cll)), 'Color',grayColor); hold on
                success_dff_per_trial(trind,:, cll) = normalize(dff_bin(:,cll));                
            end
        end
        
        meandff = squeeze(mean(success_dff_per_trial, 1, 'omitnan'));
        for cll=1:size(dff,1) % mean trial plot each cell
            subplot(ceil(sqrt(size(dff,1))),ceil(sqrt(size(dff,1))),cll)
            plot(meandff(:,cll), 'k', 'LineWidth', 2); hold on
            xline((rewloc-rewsize/2)/bin_size, 'b--', 'LineWidth', 1.5) % mark beginning of reward zone
            ylabel('dFF')
            xticks(0:5:nbins)
            xticklabels(0:45:track_length)
            yyaxis right
            plot(vel_bin, 'Color', purple, 'LineWidth',2); 
            ylabel('Velocity (cm/s)')
            xlabel('Position (cm)')            
        end
        sgtitle(sprintf('failed trials \n plane %i, epoch %i', pln, ep))        
        pptx.addPicture(fig);
        % export_fig(fullfile(savedst, sprintf('%s_failed_trials_cell_profiles_pln%i_ep%i', an, pln, ep)), '-jpg')
        close(fig)
        end
    end
end

% save ppt
fl = pptx.save(fullfile(savedst,sprintf('%s_failed_trial_cell_profiles',an)));
