clearvars
Settings.fmatpaths=dir('Z:\sstcre_imaging\e201\**\Fall.mat');
for i=1:length(Settings.fmatpaths)
    [parent,nm,~] = fileparts(fileparts(fileparts(fileparts(Settings.fmatpaths(i).folder))));
    dy{i} = str2num(nm);
end
dys = cell2mat(dy); % some days may have beahvior but not imaging and vice versa
Settings.hrz_days = [50:60]; % get days after this day; ZD added for her folder structure
Settings.fmatpaths = Settings.fmatpaths(ismember(dys,Settings.hrz_days)); % only certain days

Settings.vrpaths=dir('Z:\sstcre_imaging\e201\**\behavior\vr\*.mat');
for i=1:length(Settings.vrpaths)
    [parent,nm,~] = fileparts(fileparts(fileparts(Settings.vrpaths(i).folder)));
    dy{i} = str2num(nm);
end
dys = cell2mat(dy);
Settings.vrpaths = Settings.vrpaths(ismember(dys,Settings.hrz_days)); % only certain days

Settings.level_mouse_name = 3;
Settings.level_day = 4;
data = struct(); % if want to create a new dataset, comment line 6 and lines 14-16 (if ismember(day_cd ......)
% load('Y:\E186\E186\alldays_info.mat'); % contains a struct called data
rewrange = 10 *2/3;
for d = 1:size(Settings.fmatpaths,1)
    file = fullfile(Settings.fmatpaths(d).folder,Settings.fmatpaths(d).name);
    directory = file;
    info = split(directory,'\');
    mouse_cd = string(info{Settings.level_mouse_name});
    day_cd = string(info{Settings.level_day});

%     if ismember(day_cd, [data.day]) % if there are too many days, this 
%         continue                    % could save some time by skipping the 
%     end                             % days already in the dataset.
    
    load(file)
    load(fullfile(Settings.vrpaths(d).folder,Settings.vrpaths(d).name))
    specific_date = extractAfter(extractBefore(VR.name_date_vr, "_time"), ...
        sprintf(upper(mouse_cd)+'_'));
    data(d,1).day = day_cd;
    data(d,1).date = char(datetime(specific_date,'InputFormat','dd_MMM_yyyy','Format','yyyy-MM-dd'));
    data(d,1).mouse = mouse_cd;
    rewloc = find(changeRewLoc ~= 0);
    loc = [find(changeRewLoc ~= 0) length(changeRewLoc)];
    one_third_y_frames = find(ybinned <= 60 & ybinned ~= 1 & ybinned ~= 1.5);
    dark_frames = find(ybinned == 1);
    nodark_frames = find(ybinned~=1 & ybinned~=1.5);
    for ep = 1:length(rewloc)
        rewl = changeRewLoc(rewloc(ep));
        start_ep = rewloc(ep);
        end_ep = loc(ep+1)-1;
        trials = trialnum(start_ep+1:end_ep);
        utrials = 3:trials(end);
        probes = unique(trials(trials <= 2));
        [moving_frames, ~] = get_moving_time_V3(forwardvel, 5, 20, 50);
        success = [];
        opto = [];
        licksnum = [];
        mean_speed = []; % how about probes?
        mean_speed_no_stop = [];
        mean_speed_one_third = [];
        mean_speed_dark = [];
        for t = 1:length(utrials)
            trial_frames = find(trialnum == utrials(t));
            trial_frames = trial_frames(ismember(trial_frames, start_ep:end_ep));
            trial_nodark_frames = trial_frames(ismember(trial_frames, nodark_frames));
            trial_moving_frames = trial_nodark_frames(ismember(trial_nodark_frames, moving_frames));
            one_third_trial_frames = trial_frames(ismember(trial_frames, one_third_y_frames));
            trial_dark_frames = trial_frames(ismember(trial_frames, dark_frames));
            if length(unique(ybinned(trial_frames))) == 1
                utrials(t) = [];
                continue
            elseif max(unique(ybinned(trial_frames))) < rewl-rewrange/2
                utrials(t) = [];
                continue
            end
            licksnum = [licksnum sum(licks(trial_frames)==1)];
            mean_speed = [mean_speed mean(forwardvel(trial_nodark_frames))];
            mean_speed_no_stop = [mean_speed_no_stop mean(forwardvel(trial_moving_frames))];
            mean_speed_one_third = [mean_speed_one_third mean(forwardvel(one_third_trial_frames))];
            mean_speed_dark = [mean_speed_dark mean(forwardvel(trial_dark_frames))];
            if ~isempty(find(rewards(trial_frames) == 1))
                success = [success 1];
            else
                success = [success 0];
            end
            if isfield(VR,'optotrigger')
                opto_index = find(VR.optotrigger == 1);
                opto_index = opto_index(ismember(opto_index, trial_frames));
                if isempty(opto_index) || length(opto_index) < 0.7*length(trial_frames)
                    opto = [opto 0];
                else
                    opto = [opto 1];
                end
            else
                opto = [opto 0];
            end
        end
        probe = [];
        pb_licks = [];
        pb_opto = [];
        for p = 1:length(probes)
            probe = [probe probes(p)];
            probe_frames = find(trialnum == probes(p));
            probe_frames= probe_frames(ismember(probe_frames, start_ep:end_ep));
            pb_licks = [pb_licks sum(licks(probe_frames)==1)];
            if isfield(VR,'optotrigger')
                opto_index = find(VR.optotrigger == 1);
                opto_index = opto_index(ismember(opto_index, probe_frames));
                if isempty(opto_index) || length(opto_index) < 0.7*length(probe_frames)
                    pb_opto = [pb_opto 0];
                else
                    pb_opto = [pb_opto 1];
                end
            else
                pb_opto = [pb_opto 0];
            end
            
        end

        data(d,1).day_eps(ep,1).epoch = ep;
        data(d,1).day_eps(ep,1).RewLoc = rewl;
        data(d,1).day_eps(ep,1).success_info = success;
        data(d,1).day_eps(ep,1).opto_stim = opto;
        data(d,1).day_eps(ep,1).licks = licksnum;
        
        data(d,1).day_eps(ep,1).mean_speed = mean_speed;
        data(d,1).day_eps(ep,1).mean_speed_no_stop = mean_speed_no_stop;
        data(d,1).day_eps(ep,1).mean_speed_one_third_track = mean_speed_one_third;
        data(d,1).day_eps(ep,1).mean_speed_in_dark = mean_speed_dark;

        data(d,1).day_eps(ep,1).probe_trial = probe;
        data(d,1).day_eps(ep,1).probe_licks = pb_licks;
        data(d,1).day_eps(ep,1).probe_opto = pb_opto;
    
    end
    if isfield(VR, 'optotrigger')
        data(d,1).Opto = 1;
    else
        data(d,1).Opto = 0;
    end
end

save('Z:\hrz_info.mat','data')
disp('All done!')