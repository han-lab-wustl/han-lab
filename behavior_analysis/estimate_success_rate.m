% look at probes the day after in hrz to create a null distribution of
% licks
% zahra - 1/24/2024
% for ed's dopamine grant
clear all;close all;
fld = "\\storage1.ris.wustl.edu\ebhan\Active\dzahra\VR_data_ZD_memory_probes";
dirs = dir(fld);
grayColor = [.7 .7 .7]; 
toplot=false;
rates = {};
for d=1:length(dirs)
    clearvars -except days grayColor d COMlick_rewlocprev lickpos fld dirs rewloc_prevs toplot sessions rates
    if contains(dirs(d).name, ").mat") && (dirs(d).name(1)=='e' || dirs(d).name(1)=='E') && ...
            dirs(d).name(5)=='_'% real mat files
        behfl = fullfile(dirs(d).folder, dirs(d).name);        
        load(behfl)
    
        eps = find(VR.changeRewLoc);
        eps = [find(VR.changeRewLoc) length(VR.ypos)];
        eprng = eps(1):eps(2); trialnum = VR.trialNum; reward = VR.reward==1;
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
                    [success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum,reward);
                    rate=success/total_trials;
                    rates{d} = rate;                    % plot
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
rates = rates(~cellfun('isempty',rates));
rates = cell2mat(rates);
SEM = std(rates)/sqrt(length(rates));               % Standard Error
ts = tinv([0.025  0.975],length(rates)-1);      % T-Score
CI = mean(rates) + ts*SEM;                      % Confidence Intervals
figure; 
plot(1, rates, 'ko', ...
    'MarkerSize',8); hold on;
boxplot(rates)
ylabel("Success Rate")