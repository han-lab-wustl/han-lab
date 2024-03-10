% zahra
% get activity of tracked cells per day
clear all;
load("Y:\analysis\celltrack\e216_daily_tracking_plane0\Results\commoncells_once_per_week.mat")
src = "Y:\analysis\fmats";

cc=commoncells_once_per_week;
mouse_name = 'e218';
dys = [20,21,22,23,35,36,37,38,39,40,41,...
     42,43,44,45,47 48 49 50];
tracked_dys = [1 2 3 4 16:26 28:31]; % corresponding to days analysing
opto_ep = [-1 -1 -1 -1 3 0 1 2 0 1 3 0 1 2 0 3 0 1 2];

mouse_name = 'e216';
dys = [37 38 39 40 41 42 43 44 45 46 47 48 50:53, 55:63, 65];
tracked_dys = [5:16,18:21, 23:32];
opto_ep = [2 -1 0 1 3 -1 -1 -1 0 1 1 2 3 0 1 2 0 1 2 0 1 3 0 2 0 2]; % corresponding to days analysing
%%
for dy=1:length(dys)
    clearvars -except dy src mouse_name dys cc opto_ep tracked_dys
    daypth = dir(fullfile(src, mouse_name, 'days', sprintf('%s_day%03d*.mat', mouse_name, dys(dy)))); 
    load(fullfile(daypth.folder,daypth.name));
    % if exist('tracked_cells_dff_fc3_table', 'var')~=1
    pcs = reshape(cell2mat(putative_pcs), [length(putative_pcs{1}), length(putative_pcs)]);
    tracked_cells = cc(:,tracked_dys(dy));    
    % get activity on opto ep
    optoep = opto_ep(dy);
    if optoep<2
        optoep=2;
    end
    try
        gainf = 1/VR.scalingFACTOR;
    catch
        gainf = 3/2; % 3/2 VS. 1; in this pipeline the gain is multiplied everywhere
    end
    rewsize = VR.settings.rewardZone*gainf;
    ybinned = ybinned*gainf; rewlocs = changeRewLoc(changeRewLoc>0)*(gainf); 
    eps = find(changeRewLoc>0);
    eps = [eps length(changeRewLoc)];      
    eprng = eps(optoep):eps(optoep+1); 
    eprng = eprng(ybinned(eprng)<rewlocs(optoep)-rewsize); % pre reward only
    tracked_cell_iind = 1:length(tracked_cells);
    tracked_cells_this_day_iind = tracked_cell_iind(tracked_cells>0); % important var to save
    dff_tracked_cells = dFF(eprng,  tracked_cells(tracked_cells>0));
    fc3_tracked_cells = Fc3(:,  tracked_cells(tracked_cells>0));
    mean_dff_per_cell_opto = mean(dff_tracked_cells,1,'omitnan');
     [mean_length_of_transients_per_cell_opto,....
    mean_length_of_transients_per_cell_ctrl, auc_transients_per_cell_opto,...
    auc_transients_per_cell_ctrl, peak_transients_per_cell_opto, ...
    peak_transients_per_cell_ctrl] = get_transient_stats(changeRewLoc, gainf, ybinned, fc3_tracked_cells,optoep);
    % previous ep
    optoep = optoep-1;
    eprng = eps(optoep):eps(optoep+1);    
    eprng = eprng(ybinned(eprng)<rewlocs(optoep)-rewsize); % compare to pre-reward of previous epoch
    tracked_cell_iind = 1:length(tracked_cells);
    tracked_cells_this_day_iind = tracked_cell_iind(tracked_cells>0); % important var to save
    dff_tracked_cells = dFF(eprng,  tracked_cells(tracked_cells>0));
    mean_dff_per_cell_prev = mean(dff_tracked_cells,1,'omitnan');
    tracked_cells_dff_fc3(1,:) = tracked_cells_this_day_iind;
    tracked_cells_dff_fc3(2,:) = mean_dff_per_cell_prev;
    tracked_cells_dff_fc3(3,:) = mean_dff_per_cell_opto;       
    tracked_cells_dff_fc3(4,:) = auc_transients_per_cell_ctrl;    
    tracked_cells_dff_fc3(5,:) = auc_transients_per_cell_opto;    

    tracked_cells_dff_fc3 = tracked_cells_dff_fc3';
    tracked_cells_dff_fc3_table = array2table(tracked_cells_dff_fc3, ...
        'VariableNames', {'tracked_cell_index', 'mean_dff_per_cell_prev', 'mean_dff_per_cell_opto', 'auc_transients_per_cell_ctrl', ...
        'auc_transients_per_cell_opto'});
    save(fullfile(daypth.folder,daypth.name), 'tracked_cells_dff_fc3_table', '-append')
    disp(fullfile(daypth.folder,daypth.name))
    % end
    % pciind = 1:size(dFF,2);
    % pc = logical(iscell(:,1));
    % pciind = pciind(pc);
    % bordercells_pc = bordercells(pc); % mask border cells
    % pciind = pciind(~bordercells_pc);
    % tracked_cells_this_day_iind_pcs = pciind(ismember(pciind,tracked_cells_this_day_iind)); % place cell indicies
end
%%
% compile tracked cell activity across days
tracked_cells_dff_fc3_tables = {};
for dy=1:length(dys)
    daypth = dir(fullfile(src, mouse_name, 'days', sprintf('%s_day%03d*.mat', mouse_name, dys(dy)))); 
    load(fullfile(daypth.folder,daypth.name), 'tracked_cells_dff_fc3_table');
    tracked_cells_dff_fc3_tables{dy} = tracked_cells_dff_fc3_table;
    disp(fullfile(daypth.folder,daypth.name));
end
%%
% plot each cell in prev vs. opto ep
opto_tracked = tracked_cells_dff_fc3_tables;%(opto_ep>1);
tracked_iind = tracked_cells_dff_fc3_tables{1}.tracked_cell_index; 
mat = {}; idx = 1;
for iind=1:length(tracked_iind)
try
opto_mean_auc_across_sessions = cellfun(@(x) x(x.tracked_cell_index==tracked_iind(iind),'auc_transients_per_cell_opto'), opto_tracked, 'UniformOutput',false);
emptyind = cell2mat(cellfun(@isempty, opto_mean_auc_across_sessions, 'UniformOutput',false));
opto_mean_auc_across_sessions(emptyind) = {array2table([NaN], 'VariableNames', {'auc_transients_per_cell_opto'})}; % pad nans
% fix format
opto_mean_auc_across_sessions =cell2mat(cellfun(@(x) x.auc_transients_per_cell_opto, opto_mean_auc_across_sessions, 'UniformOutput',false));

ctrl_mean_auc_across_sessions = cellfun(@(x) x(x.tracked_cell_index==tracked_iind(iind),'auc_transients_per_cell_ctrl'), opto_tracked, 'UniformOutput',false);
emptyind = cell2mat(cellfun(@isempty, ctrl_mean_auc_across_sessions, 'UniformOutput',false));
ctrl_mean_auc_across_sessions(emptyind) = {array2table([NaN], 'VariableNames', {'auc_transients_per_cell_ctrl'})}; % pad nans
% fix format
ctrl_mean_auc_across_sessions = cell2mat(cellfun(@(x) x.auc_transients_per_cell_ctrl, ctrl_mean_auc_across_sessions, 'UniformOutput',false));
mat{iind,1} = ctrl_mean_auc_across_sessions;
mat{iind,2} = opto_mean_auc_across_sessions;
mat{iind,3} = ctrl_mean_auc_across_sessions-opto_mean_auc_across_sessions;
idx = idx+1;
catch
    fprintf('cell %i likely not tracked all days \n', iind)
    mat{iind,1} = ones(1, length(tracked_cells_dff_fc3_tables))*NaN;
    mat{iind,2} = ones(1, length(tracked_cells_dff_fc3_tables))*NaN;
    mat{iind,3} = ones(1, length(tracked_cells_dff_fc3_tables))*NaN;
end
end
%%
mat_ = cell2mat(mat);
ll = length(mat{1,1});
diffmat = mat_(:,end-ll+1:end);
diffmat_mean = mean(diffmat,2,'omitnan');
[~,sortidx] = sort(diffmat_mean, 'descend');
diffmat_mean = diffmat_mean(sortidx,:);
mat_sort = mat_(sortidx,:);
%%
figure; imagesc(normalize(smoothdata(mat_sort(:,1:end-ll),15))); colormap jet
xline(ll, 'w--')
ylabel('Cells')
xlabel('Epoch / Session')
%%
% only opto
optomat = mat_sort(:,[(opto_ep>1) (opto_ep>1) (opto_ep>1)]);
dim_opto =size(optomat,2)/3;
diffmatopto = optomat(:,end-dim_opto+1:end);
[~,sortidx]=sort(mean(diffmatopto,2,'omitnan'), 'descend');

ctrlmat = mat_sort(:,[(opto_ep<1) (opto_ep<1) (opto_ep<1)]);
dim = size(ctrlmat,2)/3;
diffmatc = ctrlmat(:,end-dim+1:end);
%%
figure;
subplot(1,2,1);
imagesc(normalize(smoothdata(optomat(sortidx,1:dim_opto*2),15))); colormap jet
xline(dim_opto+0.5, 'w--')
ylabel('Cells')
xlabel('Epoch / Session')
title('Opto Sessions')
% vs. ctrl
subplot(1,2,2);
% sum((sum(diffmatc>0,2,'omitnan')>dim/2))/size(ctrlmat,1) % cells increased
imagesc(normalize(smoothdata(ctrlmat(sortidx,1:dim*2),15))); colormap jet
xline(dim+0.5, 'w--')
ylabel('Cells')
xlabel('Epoch / Session')
title('Control Sessions')
sgtitle('Ordered by max difference in opto sessions')
%%
iind_to_test = 1:length(diffmatopto);
iind_to_test = iind_to_test(sum(diffmatopto>0,2,'omitnan')>=3); % ctrl is higher
save('X:\vip_tracked_cells_diff_pos_e218.mat', 'iind_to_test')
load('X:\vip_tracked_cells_diff_pos_e218.mat')
% for ii=iind_to_test%size(mat_,1)
%     prev = optomat(ii,1:dim_opto); opto = optomat(ii,dim_opto+1:dim_opto+dim_opto);
%     figure; plot(1,prev, 'ko'); hold on; plot(2,opto, 'ro');
%     xlim([0 3])
%     for j=1:length(optomat(ii,1:dim))
%         plot([1 2], [prev(j), opto(j)],'k--')
%     end
%     title(sprintf('Tracked cell no %i', ii))
% end
% yticks(1:length(sortidx)); yticklabels(sortidx)
%%
% TDO: fix com median code to prevent nans
% TODO: look at cells with coms 40 cm pre-reward and compare
close all
mouse_name = 'e216';
pptx    = exportToPPTX('', ... % make new file
    'Dimensions',[12 6], ...
    'Title','tuning curves of decreased activity cells', ...
    'Author','zahra', ...
    'Subject','Automatically generated PPTX file', ...
    'Comments','This file has been automatically generated by exportToPPTX');

savedst = 'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\figure_data'; % where to save ppt of figures
tuning_curves_tracked_cells = {};
coms_tracked_cells = {};
% get tuning curves of tracked cells
for dy=1:length(dys)
    clearvars -except dy src mouse_name dys cc opto_ep tracked_dys iind_to_test savedst pptx
    daypth = dir(fullfile(src, mouse_name, 'days', sprintf('%s_day%03d*.mat', mouse_name, dys(dy)))); 
    load(fullfile(daypth.folder,daypth.name));
    pcs = reshape(cell2mat(putative_pcs), [length(putative_pcs{1}), length(putative_pcs)]);
    tracked_cells = cc(:,dy);
    tracked_cells_diff = tracked_cells(iind_to_test);
    fc3_tracked = ones(size(dFF,1), length(tracked_cells_diff))*NaN; dff_tracked = ones(size(dFF,1), length(tracked_cells_diff))*NaN;
    for ii=1:size(fc3_tracked,2)
        try
            fc3_tracked(:,ii)=Fc3(:,tracked_cells_diff(ii));
            dff_tracked(:,ii)=dFF(:,tracked_cells_diff(ii));
        catch
        end
    end
    bin_size = 3; % cm
    try
        gainf = 1/VR.scalingFACTOR;
    catch
        gainf = 3/2; % 3/2 VS. 1; in this pipeline the gain is multiplied everywhere
    end
    track_length = 180*gainf;
    try
        rew_zone = VR.settings.rewardZone*gainf; % cm
    catch
        rew_zone = 15;
    end
    % zahra hard coded to be consistent with the dopamine pipeline
    thres = 5; % 5 cm/s is the velocity filter, only get
    % frames when the animal is moving faster than that
    ftol = 10; % number of frames length minimum to be considered stopped
    ntrials = 5; % e.g. last 8 trials to compare    
    plns = [0]; % number of planes
    Fs = 31.25/length(plns);
    eps = find(changeRewLoc>0);
    eps = [eps length(changeRewLoc)];    
    rewlocs = changeRewLoc(changeRewLoc>0)*(gainf);
    
    [tuning_curves, coms, ~, ~] = make_tuning_curves(eps, trialnum, rewards, ybinned, gainf, ntrials,...
    licks, forwardvel, thres, Fs, ftol, bin_size, track_length, fc3_tracked, dff_tracked);
    slideId = pptx.addSlide();
    fprintf('Added slide %d\n',slideId);
    fig = figure('Renderer', 'painters', 'Position', [10 10 1050 800]);
    for ep=1:length(eps)-1
        subplot(1,length(eps)-1,ep)
        plt = tuning_curves{ep};
        % sort all by ep 1
        [~,sorted_idx] = sort(coms{1});
        plt = plt(sorted_idx,:);
        imagesc(normalize(plt,2));
        hold on;
        % plot rectangle of rew loc
        % everything divided by 3 (bins of 3cm)
        rectangle('position',[ceil(rewlocs(ep)/bin_size)-ceil((rew_zone/bin_size)/2) 0 ...
            rew_zone/bin_size size(plt,1)], ... 
            'EdgeColor',[0 0 0 0],'FaceColor',[1 1 1 0.5])
        colormap jet
        xticks([0:bin_size:ceil(track_length/bin_size)])
        xticklabels([0:bin_size*bin_size:track_length])
        title(sprintf('epoch %i', ep))
    end
    sgtitle(sprintf(['animal %s, day %i \n opto ep %i'], mouse_name, dys(dy), opto_ep(dy)))
    pptx.addPicture(fig);        
    comparisons = nchoosek(1:sum(cellfun(@(x) ~isempty(x),tuning_curves)),2);
    rewloccomp = zeros(size(comparisons,1),2); rewzonecomp = zeros(size(comparisons,1),2);
    for i=1:size(comparisons,1)
    comparison = comparisons(i,:);
    slideId = pptx.addSlide();
    fprintf('Added slide %d\n',slideId);
    fig = figure('Renderer', 'painters');
    plot(coms{comparison(1)}, coms{comparison(2)}, 'ko'); hold on;
    xline(rewlocs(comparison(1)), 'r', 'LineWidth', 3);
    yline(rewlocs(comparison(2)), 'r', 'LineWidth', 3)
    plot([0:track_length],[0:track_length], 'k', 'LineWidth',2)
    xlim([0 track_length]); ylim([0 track_length])
    xlabel(sprintf('ep%i', comparison(1)));
    ylabel((sprintf('ep%i', comparison(2))))
    title(sprintf(['COM (median) \n ' ...
        'animal %s, day %i,' ...
        'comparison: ep%i vs ep%i'], mouse_name, dy, comparison(1), comparison(2)))
    pptx.addPicture(fig);
    end
    % save var
    tuning_curves_tracked_cells_pos{dy} = tuning_curves;
    coms_tracked_cells_pos{dy} = coms;
    save(fullfile(daypth.folder,daypth.name), 'tuning_curves_tracked_cells_pos', 'coms_tracked_cells_pos', '-append');   
end
fl = pptx.save(fullfile(savedst,sprintf('%s_tuning_curves_tracked_cells_increased_inopto',mouse_name)));