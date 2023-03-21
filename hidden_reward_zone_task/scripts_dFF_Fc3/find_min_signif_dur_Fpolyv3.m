function std_sig_dur_min_day = find_min_signif_dur_Fpolyv3(Fc2)
%do for each day
% clear all;
close all;
disp('calculating Fc3')
tic
daycount=0;
numfiles=1;

%made into a function for


%load F and movie files, get Fname for Fpoly not
%mpgfiles

for i=1:numfiles
    
    %load F file
    %     [Ffile,Ffilepath]=uigetfile('*F.mat','pick the F file');
    %     fullFfile=[Ffilepath Ffile];
    %     load(fullFfile);
    
    daycount=daycount+1;
    %     if exist('Fca','var')
    %         Fc2=full(Fca');
    %     end
    
    
    %     skew_cutoff = 2 ;
    Fc2std=zeros(size(Fc2));
    Fc2state=zeros(size(Fc2));
    negFc2state=zeros(size(Fc2));
    numtraces=size(Fc2,2);
    %     putative_interneurons = find(skewness(Fc2)<skew_cutoff);
    %     all_cells = 1:numtraces;
    %     putative_pyrs = all_cells;
    %     putative_pyrs(putative_interneurons) = [];
    
    %     upperbase(i)
    %define baseline, find start, duration and max std amplitude of
    %transients in both positive and neg. direction.
    for i=1:numtraces
        upperbase(i)=2*std(Fc2(:,i)); % noise threshold %% EB i was '1'
        baselineind=find(Fc2(:,i)<upperbase(i)); %indxes for baseline
        baseline=Fc2(baselineind,i); % grab all the baseline values for statistics
        basemedian=median(baseline);
        basestd=std(baseline);
        Fc2std(:,i)=(Fc2(:,i)-basemedian)/basestd; % bring the baseline around (0; all values == median), make a standardized Fc2
        %positive going transients
        Fc2state(:,i)=double_thresh(Fc2std(:,i),2.0,0.5); % binary vector defining where the signal goes from 0.5 to 2 std
        upticks=find(diff(Fc2state(:,i))==1); % when transient starts
        downticks=find(diff(Fc2state(:,i))==-1); % when transients stop
        %deal with end cases
        if numel(upticks)>0
            if numel(upticks) == 1 && numel(downticks) == 0
                downticks(1) = size(Fc2std(:,i),1);
            elseif numel(downticks) == 1 && numel(upticks) == 0
                upticks(1) = 1;
            end
            if upticks(1)>downticks(1)
                upticks=[1 upticks'];
            end      
            if upticks(end)>downticks(end)
                downticks=[downticks' size(Fc2,1)];
            end
        else
            transdur(i,1)=0;
            transmax(i,1)=0;
            transupticks(i,1)=0;
        end
        for j=1:length(upticks) %for every transient...
            if  isnan(mean(Fc2(:,size(Fc2,2)))) && i==numtraces
                transmax(i,j)=0; %find max std dev away from baseline of transient
                transdur(i,j)=0; %find index length of transient
                transupticks(i,j)=0;
            else
                transmax(i,j)=max(Fc2std(upticks(j):downticks(j),i)); %find max std dev away from baseline of transient
                transdur(i,j)=downticks(j)-upticks(j); %find index length of transient
                transupticks(i,j)=upticks(j); %find starting point of transients for each cell
            end
        end
        
        numframes=size(Fc2,1);
        %negative going transients
        negFc2state(:,i)=double_thresh(-Fc2std(:,i),2.0,0.5);
        negupticks=find(diff(negFc2state(:,i))==1);
        negdownticks=find(diff(negFc2state(:,i))==-1);
        if numel(negupticks)>0
            if negupticks(1)>negdownticks(1)
                negupticks=[1 negupticks'];
            end
            if negupticks(end)>negdownticks(end)
                negdownticks=[negdownticks' numframes];
            end
        else
            negtransmax(i,1)=0; %edit mwa orginal: negtransmax(i)=0; was giving errors
            negtransdur(i,1)=0;
            negtransupticks(i,1)=0;
        end
        for j=1:length(negupticks)
            if  isnan(mean(Fc2(:,size(Fc2,2)))) && i==numtraces
                negtransmax(i,j)=0; %find max std dev away from baseline of transient
                negtransdur(i,j)=0; %find index length of transient
                negtransupticks(i,j)=0;
            else
                negtransmax(i,j)=max(-Fc2std(negupticks(j):negdownticks(j),i));
                negtransdur(i,j)=negdownticks(j)-negupticks(j);
                negtransupticks(i,j)=negupticks(j);
            end
            if  length(negupticks) == 0 && i~=1 && exist('negtransdur','var')
                negtransmax(i,:)=NaN(size(negtransmax(i-1,:)));
                negtransdur(i,:)=NaN(size(negtransdur(i-1,:)));
                negtransupticks(i,:)=NaN(size(negtransupticks(i-1,:)));
            end
        end
    end
    
    
    %find ratio of number of positive to neg transients for different stds
    %and of varying duration
    
    X=[0 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 inf]; % trans duration we want to study
    for k=2:5 %std from baseline
        if k==5
            n=(transmax>k);
            nn=transdur.*n;
            m=(negtransmax>k);
            %             negtransdur=zeros(size(m)); %mwa hacky
            mm=negtransdur.*m;
            
        else
            n=(and(transmax<(k+1),transmax>k));% logical matrix of transients within this range
            nn=transdur.*n; % keep only transient durantion in the std range we are interested in
            m=(and(negtransmax<(k+1),negtransmax>k)); % % logical matrix of neg transients within this range
            %              negtransdur=zeros(size(m)); %mwa hacky
            mm=negtransdur.*m; %make all zero.
            
        end
        %find significant transients by only looking at neurons
        nnhold=nn(1:end,:);
        mmhold=mm(1:end,:);
        for i = 1:numtraces
            histtrans{i}(k,1:size(X,2))=histc(nnhold(i,:),X); % count for each std (k) how many transients have a certain length
            % NB '0' means how many transients have a duration of <2 frames
            histnegtrans{i}(k,1:size(X,2))=histc(mmhold(i,:),X); % count how many negative transients (how many zeros)
            ratio_hist{i}=histnegtrans{i}./histtrans{i};
            
        end
        
        
        
        
        
    end
    
    %plot(ratio_hist');
    
    %sum all all files for each day
    if daycount==1
        histtrans_day=histtrans;
        histnegtrans_day=histnegtrans;
    else
        histtrans_day=histtrans_day+histtrans;
        histnegtrans_day=histnegtrans_day+histnegtrans;
    end
    
end


ratio_hist_day=ratio_hist;
% figure;
% for i = 1:numtraces
%     
%     plot(X,ratio_hist_day{i}'); % proportion of negative transients respect to all the others for each std studied
%     %plot(X(2:end),ratio_hist_day(:,2:end)');
%     pause(0.01)
% end
% 


%find minimum duration for a transient of each std to be significant (here
%defined as error rate of <5%)
for i = 1:numtraces
    for k=2:5
        if size((find(ratio_hist_day{i}(k,:)<.05)),2)>0
            std_sig_dur_min_day(i,k)=min(X(find(ratio_hist_day{i}(k,:)<.05)));
        else
            std_sig_dur_min_day(i,k)=inf;
        end
    end
end

toc

