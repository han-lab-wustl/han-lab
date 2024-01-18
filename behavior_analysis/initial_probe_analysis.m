% look at probes the day after
% days = [60:74 76 78:90]; % e200
% days = [50:80,82:94]; % e201
% days = [1 4:7 9:12]; % e145
% days = [67 70 73 76 80 83 86 89]; % only ctrl days after opto ep3 for e200
% days = [57 60 63 66 69 72 76 79 85 90 94];% only ctrl days after opto ep3 for e200
% fld = "Z:\sstcre_imaging\e201";
% fld = "H:\E145";
% fld = "Y:\sstcre_imaging\e200";
clear all;close all;
fld = "D:\adina_vr_files\VR_data_ZD_memory_probes";
dirs = dir(fld);
grayColor = [.7 .7 .7]; 
COMlick_rewlocprev={}; lickpos = {}; rewloc_prevs={};
for d=1:length(dirs)
    clearvars -except days grayColor d COMlick_rewlocprev lickpos fld dirs rewloc_prevs    
    if contains(dirs(d).name, ").mat") && (dirs(d).name(1)=='e' || dirs(d).name(1)=='E') && ...
            dirs(d).name(5)=='_'% real mat files
        behfl = fullfile(dirs(d).folder, dirs(d).name);        
        load(behfl)
    
%     % for 
%     behfl = dir(fullfile(fld, sprintf("Day%i",days(dd)), "E145*.mat"));
%     load(fullfile(behfl.folder, behfl.name))
    % check if probes in the beginning of the day
    % plot only first probes
        eps = find(VR.changeRewLoc);
        eps = [find(VR.changeRewLoc) length(VR.ypos)];
        eprng = eps(1):eps(2); trialnum = VR.trialNum;
        % check if initial probes exist
        if length(eprng(trialnum(eprng)<3))>10             
            % calc COM lick
            proberng = eprng(trialnum(eprng)<3); %range for initial probes
            pos = VR.ypos(proberng); % include scale factor
            trialnumep = trialnum(proberng);
            lick = VR.lick(proberng);
            COMlick = mean(pos(lick>0));            
            % get prev day rewloc
            s = dirs(d).name(6:16);
            t = datetime(s,'InputFormat','dd_MMM_yyyy');
            t.Format = 'dd_MMM_yyyy'; % format
            prev = strcat(dirs(d).name(1:5), string(t-1), '*');
            prev_fls = dir(fullfile(dirs(d).folder, prev));
            % if there are multiple files, take last one (likely started
            % and restarted)
            if length(prev_fls)>0
                prev_fl = prev_fls(end);
        %         behfl = dir(fullfile(fld, sprintf("Day%i",days(dd-1)), "E145*.mat"));
                VR_prev = load(fullfile(prev_fl.folder, prev_fl.name));
                rewloc_prev = VR.changeRewLoc(VR.changeRewLoc>0); % all epochs
                if length(rewloc_prev)>3 %previous day should have at least 3 ep
                    COMlick_rewlocprev{d} = COMlick - rewloc_prev;
                    rewloc_prevs{d} = rewloc_prev;
                    lickpos{d} = pos(lick>0);
                    % plot
                    figure;
                    plot(VR.ypos, 'Color', grayColor); hold on; 
                    plot(VR.changeRewLoc, 'b')
                    plot(find(VR.lick),VR.ypos(find(VR.lick)),'r.')     
                    rew = VR.reward>0.5; % codes for single or double rewards
                        
                    rectangle('position',[min(eprng(trialnum(eprng)<3)) 0 ...
                            length(eprng((trialnum(eprng)<3))) 180], ...
                                'EdgeColor',[0 0 0 0],'FaceColor',[0 0 1 0.3])
                    title(sprintf("%s", dirs(d).name))
                end
            end
        end
    end    
end
%%
% histogram of lick pos
lickpos = lickpos(~cellfun('isempty',lickpos));
figure;
for li=1:length(lickpos)    
    histogram(lickpos{li}, 'FaceAlpha', 0.2); hold on;
end
% rewloc distribution for each epoch
grayColor2 = [.4 .4 .4];
rewloc_prevs = rewloc_prevs(~cellfun('isempty',rewloc_prevs));
ep1 = cell2mat(cellfun(@(x) x(1), rewloc_prevs, 'UniformOutput',false));
ep2 = cell2mat(cellfun(@(x) x(2), rewloc_prevs, 'UniformOutput',false));
ep3_mask = cell2mat(cellfun(@length, rewloc_prevs, 'UniformOutput',false))>=3;
ep3 = cell2mat(cellfun(@(x) x(3), rewloc_prevs(ep3_mask), 'UniformOutput',false));
figure; plot(ep1,'k*'); hold on; plot(ep2,"b*"); plot(ep3,"r*")
legend('ep1','ep2','ep3')
%%
% check similarity to ep 1,2,3
grayColor2 = [.4 .4 .4];
COMlick_rewlocprev = COMlick_rewlocprev(~cellfun('isempty',COMlick_rewlocprev));
figure; 
plot(1, cell2mat(cellfun(@(x) x(1), COMlick_rewlocprev, 'UniformOutput',false)), 'ko', ...
    'MarkerSize',8); hold on;
plot(2, cell2mat(cellfun(@(x) x(2), COMlick_rewlocprev, 'UniformOutput',false)), ...
    'Color', grayColor2,'Marker','o', 'MarkerSize',8)
% only apply to days where more than 3 epochs
ep3_mask = cell2mat(cellfun(@length, COMlick_rewlocprev, 'UniformOutput',false))>=3;
plot(3, cell2mat(cellfun(@(x) x(3), COMlick_rewlocprev(ep3_mask), 'UniformOutput',false)), ...
    'Color', grayColor,'Marker','o','MarkerSize',8)
lastep_mask = cell2mat(cellfun(@length, COMlick_rewlocprev, 'UniformOutput',false));
lastep_COM = {};
for ii=1:length(COMlick_rewlocprev)
    ddd = cell2mat(COMlick_rewlocprev(ii));
    disp(ddd)
    lastep_COM{ii} = ddd(lastep_mask(ii));
end
plot(4, cell2mat(lastep_COM(~cellfun('isempty',lastep_COM))), ...
    'Color', 'b','Marker','o','MarkerSize',8)

xlim([0 5])
xticks([1 2 3 4])
xticklabels([{'ep1'}, {'ep2'}, {'ep3'}, {'last ep'}])
ylabel("COM-rewloc (cm)")