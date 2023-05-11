function plot_failures(data)
% zahra adapted to data structure from get epoch info
total_failure_success_days = int16.empty;
dates = strings(1, size(data,1));
all_rewypos = zeros(size(data,1), 4);
all_epocvec = zeros(size(data,1),4);
for this_day = 1:size(data,1)

    mouse_cd = data(this_day).mouse;

    vr = data(this_day).VR;
    date = extractAfter(extractBefore(vr.VR.name_date_vr, "_time"), sprintf("%s_", mouse_cd));
    date_element = split(date, "_");
    switch(date_element{2})
        case("Dec")
        date_element{2} = "12";
        case("Jan")
        date_element{2} = "01";
        case("Nov")
        date_element{2} = "11";
        case("Oct")
        date_element{2} = "10";
    end
    specific_date = extractAfter(date_element{3}, "20")+"-"+date_element{2}+"-"+date_element{1};
    dates(this_day) = specific_date;
    
    imageSync = [];

%Find start and stop of imaging using VR


    %if isfield(VR,'imageSync') %makes sure VR has an imageSync variable, if not uses abf, BUT still uses VR variables later
    imageSync = vr.imageSync;
%     else
%         [abffilename,abfpath] = uigetfile('*.abf','pick your abf file');
%         abffullfilename = [abfpath char(abffilename)];
%         data = abfload(abffullfilename);
%         imageSync = data(:,8);
%         
%     end

    inds=find((abs(diff(imageSync))>0.3*max(abs(diff(imageSync))))==1);
    meaninds=mean(diff(inds));
    figure;subplot(2,1,1);hold on;plot(imageSync);plot(abs(diff(imageSync))>0.3*max(abs(diff(imageSync))),'r');
    % subplot(2,1,1); hold on; scatter(1000*(vr.time),zeros(1,length(vr.time)),20,'y','filled');
    subplot(2,1,2);hold on;plot(imageSync);plot(abs(diff(imageSync))>0.3*max(abs(diff(imageSync))),'r');
    xlim([inds(1)-2.5*meaninds inds(1)+2.5*meaninds]);
    input('');
    % xlim([1.064*10^4 1.078*10^4])
    [uscanstart,y]=ginput(1)
    uscanstart=round(uscanstart)

    figure;subplot(2,1,1);hold on;plot(imageSync);plot(abs(diff(imageSync))>0.3*max(abs(diff(imageSync))),'r');
    subplot(2,1,2);hold on;plot(imageSync);plot(abs(diff(imageSync))>0.3*max(abs(diff(imageSync))),'r');
    xlim([inds(end)-4*meaninds inds(end)+2*meaninds]);
    % xlim([6.92*10^4 6.94*10^4])
    input('');
    [uscanstop,y]=ginput(1)
    uscanstop=round(uscanstop)
    disp(['Length of scan is ', num2str(uscanstop-uscanstart)])
    disp(['Time of scan is ', num2str((vr.time(uscanstop)-vr.time(uscanstart)))])



    close all;
    if ~isfield(VR,'imageSync') %if there was no vr.imagesync, rewrites scanstart and scanstop to be in VR iteration indices
        %         buffer = diff(data(:,4))
        VRlastlick = find(vr.lick>0,1,'last');
        abflicks = findpeaks(-1*data(:,3),5);
        buffer = abflicks.loc(end)/1000-vr.time(VRlastlick);
        check_imaging_start_before = (uscanstart/1000-buffer); %there is a chance to recover imaging data from before you started VR (if you made an error) in this case so checking for that
        [trash,scanstart] = min(abs(vr.time-(uscanstart/1000-buffer)));
        [trash,scanstop] = min(abs(vr.time-(uscanstop/1000-buffer)));
    else
        scanstart = uscanstart;
        scanstop = uscanstop;
        check_imaging_start_before = 0; %there is no chance to recover imaging data from before you started VR so sets to 0
        
    end


%cuts all of the variables from VR
    urewards=vr.reward(scanstart:scanstop); 
    uimageSync=imageSync(scanstart:scanstop); 
    uforwardvel=-0.013*vr.ROE(scanstart:scanstop)./diff(vr.time(scanstart-1:scanstop));  
    uybinned=vr.ypos(scanstart:scanstop);   
    unumframes=length(scanstart:scanstop);
    uVRtimebinned = vr.time(scanstart:scanstop)- check_imaging_start_before-vr.time(scanstart);
    utrialnum = vr.trialNum(scanstart:scanstop);
    uchangeRewLoc = vr.changeRewLoc(scanstart:scanstop);
    uchangeRewLoc(1) = vr.changeRewLoc(1);
    ulicks = vr.lick(scanstart:scanstop);
    ulickVoltage = vr.lickVoltage(scanstart:scanstop);

    rewrange = 10; %rewardzonesize = 10; set 15 to make sure it covers all rewards
    rewloc = find(uchangeRewLoc ~= 0);
    total_failure_success_epochs = zeros(length(rewloc), 3);
    loc = [rewloc, size(uchangeRewLoc, 2)];
    for idx = 1:length(rewloc)
        rewl = uchangeRewLoc(1, loc(idx));
        rl = find(uybinned > rewl-rewrange & uybinned < rewl+rewrange & urewards == 1);
        rl = rl(rl < loc(idx+1) & rl >= loc(idx));
        success_per_epoch = length(unique(utrialnum(rl)));
        trialn= max(utrialnum(1, loc(idx):loc(idx+1)-1));
%         if idx == length(rewloc) & sum(utrialnum(1, loc(idx):loc(idx+1)-1) == trialn & uybinned(1, loc(idx):loc(idx+1)-1) > rewl-rewrange)==1 % remove the last one without reaching the reward zone
%             trialn = trialn-1;
%         end
        if idx == length(rewloc) & trialn <= 2
            trialn = 0;
%         elseif idx > 1
%             trialn = trialn - 3;
        else %if idx == 1
            trialn = trialn-3;
        end
        failure_per_epoch = trialn - success_per_epoch;
        total_failure_success_epochs(idx,1) = failure_per_epoch;
        total_failure_success_epochs(idx,2) = success_per_epoch;
        total_failure_success_epochs(idx,3) = trialn; %failure_per_epoch + success_per_epoch;
    end
    failure_per_day = sum(total_failure_success_epochs(:,1));
    success_per_day = sum(total_failure_success_epochs(:,2));
    failure_percentage = double(total_failure_success_epochs(:,1))./double(total_failure_success_epochs(:,3));
    for p = 1:length(failure_percentage)
        if isnan(failure_percentage(p))
            failure_percentage(p) = 0;
        end
    end
    rewypos = uchangeRewLoc(rewloc);
    all_epocvec(this_day,1:length(rewypos)) = 1:length(rewypos);
    all_rewypos(this_day, 1:length(rewypos)) = rewypos;
    epochs = "ep" + string(1: length(rewloc));
    f = figure;%('visible','off');
    eps = categorical(epochs);
    yyaxis left
    bg = bar(eps, total_failure_success_epochs(:,1:2), 'stacked');
    colorss = lines(length(bg));
    for ii = 1:length(bg)
        set(bg(ii), 'FaceColor', colorss(ii,:));
    end
    ylabel("Number of failures and successes");
    hold on;
    %plot(eps, failure_percentage.*double(max(total_failure_success_epochs(:,3))),'LineWidth',2, 'Color','k','LineStyle','--');
    text(eps, total_failure_success_epochs(:,1)./2,num2str(failure_percentage.*100, '%.1f')+"%",'Color','w','HorizontalAlignment','center','FontWeight','bold');
    yyaxis right
    scatter(eps, rewypos, 100,'k','filled','o');
    ylim([0 180]);
    ylabel("Reward location (cm)");
    title("Number of Failures During Each Epoch on "+ day_cd+mouse_cd);
    legend('Failure', 'Success', 'Reward Location');
    total_failure_success_days(this_day, :) = [failure_per_day success_per_day];
    saveas(f,['H:\E186\E186\failure_fig\' char(day_cd) '_failureNum.png'],'png')
    hold off;
end

all_rewypos(all_rewypos==0) = NaN;
all_epocvec(all_epocvec==0) = NaN;
fig = figure;
x_dates = categorical(dates);
yyaxis left;
b = bar(x_dates, total_failure_success_days(:, 1:2), 'stacked');
colorss = lines(length(b));
for ii = 1:length(b)
    set(b(ii), 'FaceColor', colorss(ii,:));
end
hold on;

title("Number of Failures per Imaging Day: "+mouse_cd);
total_failure_success_days(:,3) = total_failure_success_days(:,1) + total_failure_success_days(:,2);
freq = double(total_failure_success_days(:,1))./double(total_failure_success_days(:,3));
text(x_dates, double(total_failure_success_days(:,1))./2,num2str(freq.*100, '%.0f')+"%",'Color','w','HorizontalAlignment','center');
ylabel("Number of failures and successes");
yyaxis right;
scatter(x_dates, all_rewypos, 100,'k','o','LineWidth', 1.5);
hold on;
for iii = 1:size(all_epocvec,2)
    text(x_dates, all_rewypos(:,iii), string(all_epocvec(:,iii)), 'Color','k','HorizontalAlignment','center','FontWeight','bold');
end
ylabel("Reward location (cm)");
ylim([0 180]);
legend('Failure', 'Success', 'Reward Location');
%lgd.Location = 'northeastoutside';
%saveas(fig,'H:\E186\E186\failure_fig\Total_failureNum.png','png');
hold off;
disp("done!");end