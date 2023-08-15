load('Z:\sstcre_imaging\e201\55\230502_ZD_000_001\suite2p\plane0\Fall.mat')
eps = find(changeRewLoc);
eps = [eps length(changeRewLoc)]; % includes end of recording as end of a epoch
ep = 1;    
eprng = eps(1):eps(2);
mask = trialnum(eprng)>=3; % skip probes
ypos = ybinned(eprng);
ypos = ypos(mask)*(3/2); % gain factor for zahra mice
[time_moving ,time_stop] = vr_stop_and_moving_time(ypos);
fc3 = all.Fc3(:,eprng);
fc3 = fc3(:,mask);
rewloc = changeRewLoc(changeRewLoc>0); % reward location
rewlocopto = rewloc(ep)*(3/2);
%%
% check to see if there are any negative transients in fc3
figure; plot(fc3(randi([1 size(fc3,1)],1),:))
xlabel('frames in ep1')
ylabel('fc3')
title(sprintf('cell no . %i', randi([1 size(fc3,1)],1)))
%% 
% calculate spatial info
% only during moving time?
fc3_mov = fc3(:,time_moving);
ypos_mov = ybinned(:,time_moving);
bin = 3; % cm t
track_length = 270;
spatial_info = get_spatial_info_all_cells(fc3_mov',ypos_mov,31.25, ...
    ceil(track_length/bin),track_length);

% shuffle params
%shuffle transients and not fc3 values themselves = use bwlabel
transients = zeros(size(fc3_mov));
transients_shuf = zeros([size(fc3_mov) 1000]);
for cell=1:size(fc3_mov)
%     disp(cell)
    fc3cell = fc3_mov(cell,:);
    if sum(fc3cell)>0 % check if 0 dff?
        trsnt = bwlabel(fc3cell); % gets rid of inidividual ids for transients
        for shuf=1:1000 % shuffle 1000 times
            if rem(shuf,100)==0
                disp(shuf)
            end
            i=0; spk_shufs = {};
            for spk=unique(trsnt)% shuffle each each transient
                if spk~=0
                    start = find(trsnt==spk,1); 
                    endspk = find(trsnt==spk,1,'last');
                    spkl = endspk-start; % range of spike
                    spk_shuf = zeros(size(trsnt));
                    spk_shuf(start)=1;
                    spk_shuffle=spk_shuf(randperm(length(spk_shuf))); % shuffle in time
                    spkstart_shuf = find(spk_shuffle>0);
                    if spkstart_shuf+spkl>length(spk_shuf) % make sure end of shuffle transient does not exceed length of array
                        shuf_end = length(spk_shuf);
                    else
                        shuf_end = spkstart_shuf+spkl;
                    end
                    fc3back = fc3cell(trsnt==spk); % put back fc3               
                    spk_shuffle(spkstart_shuf:shuf_end)=fc3back(1:length(spkstart_shuf:shuf_end)); % add fc3 back to shuffled transient
                    i=i+1;
                    if i==1
                        spk_shufs{i} = spk_shuffle;
                    else
                        spk_shufs{i} = spk_shuffle+spk_shufs{i-1}; %add up fc3 together
                    end
                end
            end
            transients_shuf(cell,:,shuf) = spk_shufs{length(spk_shufs)}; %only get the last one of the shuffles spikes
        end        
        transients(cell,:) = fc3cell;        
    end
end