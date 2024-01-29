% look at probes the day after in hrz to create a null distribution of
% licks
% zahra - 1/24/2024
% for ed's dopamine grant
clear all;close all;
fld = "\\storage1.ris.wustl.edu\ebhan\Active\dzahra\VR_data_ZD_memory_probes";
dirs = dir(fld);
grayColor = [.7 .7 .7]; 
toplot=false;
COMlick_rewlocprev={}; lickpos = {}; rewloc_prevs={}; sessions = {};
for d=1:length(dirs)
    clearvars -except days grayColor d COMlick_rewlocprev lickpos fld dirs rewloc_prevs toplot sessions   
    if contains(dirs(d).name, ").mat") && (dirs(d).name(1)=='e' || dirs(d).name(1)=='E') && ...
            dirs(d).name(5)=='_'% real mat files
        behfl = fullfile(dirs(d).folder, dirs(d).name);        
        load(behfl)
    
        eps = find(VR.changeRewLoc);
        eps = [find(VR.changeRewLoc) length(VR.ypos)];
        eprng = eps(1):eps(2); trialnum = VR.trialNum;
        % check if initial probes exist
        if length(eprng(trialnum(eprng)<3))>10 % at least 10 frames of probes             
            % calc COM lick
            proberng = eprng(trialnum(eprng)<3); %range for initial probes
            pos = VR.ypos(proberng)/VR.scalingFACTOR; % include scale factor
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
                rewloc_prev = VR.changeRewLoc(VR.changeRewLoc>0)/VR.scalingFACTOR; % all epochs
                if length(rewloc_prev)>3 %previous day should have at least 3 ep
                    disp(prev_fl.name)
                    COMlick_rewlocprev{d} = COMlick-rewloc_prev;
                    rewloc_prevs{d} = rewloc_prev;
                    lickpos{d} = pos(lick>0);
                    sessions{d} = behfl;
                    % plot
                    if toplot==true
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
        else
            delete(behfl) % delete behavior file if it doesn't have previous day behavior data  
        end
    end    
end
%%
% histogram of lick pos
% lickpos = lickpos(~cellfun('isempty',lickpos));
% figure;
% for li=1:length(lickpos)    
%     histogram(lickpos{li}, 'FaceAlpha', 0.2); hold on;
% end
% % rewloc distribution for each epoch
% grayColor2 = [.4 .4 .4];
% rewloc_prevs = rewloc_prevs(~cellfun('isempty',rewloc_prevs));
% ep1 = cell2mat(cellfun(@(x) x(1), rewloc_prevs, 'UniformOutput',false));
% ep2 = cell2mat(cellfun(@(x) x(2), rewloc_prevs, 'UniformOutput',false));
% ep3_mask = cell2mat(cellfun(@length, rewloc_prevs, 'UniformOutput',false))>=3;
% ep3 = cell2mat(cellfun(@(x) x(3), rewloc_prevs(ep3_mask), 'UniformOutput',false));
% figure; plot(ep1,'k*'); hold on; plot(ep2,"b*"); plot(ep3,"r*")
% legend('ep1','ep2','ep3')
%%
% check similarity to ep 1,2,3
grayColor2 = [.4 .4 .4];
grayColor2 = [.2 .2 .2];
COMlick_rewlocprev = COMlick_rewlocprev(~cellfun('isempty',COMlick_rewlocprev));
sessions = sessions(~cellfun('isempty',sessions));

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
xticklabels([{'Epoch 1'}, {'Epoch 2'}, {'Epoch 3'}, {'Last Epoch'}])
xlabel("Trial Epoch")
ylabel("Center of Mass Licks-Epoch Reward Location (cm)")