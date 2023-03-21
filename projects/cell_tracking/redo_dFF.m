function [dFF]=redo_dFF(F,Fs,varargin)
if nargin>2
    time=varargin{1};
else, time=900;
end
if nargin>3
    nF=varargin{2};
else nF=[];
end
if nargin>4
    plotOpt =1;
    figure
else, plotOpt=0;
end

% F=mouse(3).F0{14}.raw{4};
dFF=zeros(size(F));
for j=1:size(F,2)
    if ~isempty(nF)
%     correctedF=(F(:,j)-.8*nF(:,j));
        correctedF=F(:,j)-.8*demean(nF(:,j));
        if min(correctedF)<0
            correctedF=F(:,j);
        end
    else
        correctedF=F(:,j);
    end
    numframes=length(correctedF);
    window_dF=round(Fs*time);
    
     junk2=zeros(size(correctedF));   
     G=junk2;
%     smooth_correctedF=smooth(correctedF,(Fs*5));
    for k=1:round(length(correctedF))
        cut=correctedF(max(1,k-window_dF):min(numframes,k+window_dF));
        cutsort=sort(cut);
        a=round(length(cut)*.05);
        junk2(k)=cutsort(a);
%         junk2(k)=min(cut);
%         G(k)=mean(cut);
    end
%     figure('units','normalized', 'Position', [.01 .05 .98 .45]);
%     subplot(1,4,1); plot(F(:,j)); hold on; plot(nF(:,j)); legend('raw','neuro');
%     subplot(1,4,2); plot(correctedF); hold on;plot(G); legend('corF','mean')
% %     subplot(1,5,3); plot((correctedF-junk2)./junk2); legend('dF/F 5%')
%     subF=correctedF-G+mean(correctedF);
%     minSubF=min(smooth(subF,Fs*2));
%     subplot(1,4,3); plot(subF); hold on; plot(minSubF); legend('subF','minSubF');
%     subplot(1,4,4); plot((subF-minSubF)./minSubF)
    
    x=1:round(numframes);
    if ~isnan(mean(correctedF))
%                 fitData=fit((x)',junk2','exp2');
                
%                 xBase=((fitData.a)*exp((fitData.b)*(xLong))+(fitData.c)*exp((fitData.d)*(xLong)));
%                 xBase=((fitData.a)*exp((fitData.b)*(xLong)));
        xBase=junk2;

%         fitData2=fit((x)',G','exp1');
%         xBase2=((fitData2.a)*exp((fitData2.b)*(xLong))+(fitData2.c)*exp((fitData2.d)*(xLong)));
%                 xBase2=((fitData2.a)*exp((fitData2.b)*(xLong)));
%         xBase2=G;


        dFF(:,j)=((correctedF)-(xBase))./xBase;
        
%         dFF(:,j)=((subF)-(minSubF))/minSubF;
        
%         figure('units','normalized', 'Position', [.01 .05 .98 .87]);
%         subplot(2,2,1); plot(correctedF); hold on; plot(junk2'); plot(xBase');
%         subplot(2,2,2); plot(dFF(:,j))
%         
%         subplot(2,2,3); plot(correctedF); hold on; plot(G); plot(xBase2);
%         subplot(2,2,4); plot(correctedF-xBase2'+mean(correctedF))
%         d2=correctedF-xBase2'+mean(correctedF);
%         d2s=sort(smooth(d2,Fs*10));        
%         eightPer=d2s(round(.08*length(d2s)));
%         dFF2(:,j)=(d2-eightPer)/eightPer;
        
%         dFF2(:,j)=(correctedF-xBase2')./xBase2';
        if plotOpt
            nSq=ceil(sqrt(size(F,2)));
            subplot(nSq,nSq,j)
            plot(junk);
            hold on;
            plot(xBase);
        end
    end
end
    if plotOpt
        figure
        nSq=ceil(sqrt(size(F,2)));
        for j=1:size(F,2)
            subplot(nSq,nSq,j)
            plot(dFF(:,j));
        end
    end
    
    % Old Fc method
    % if nargin>2
    %     time=varargin{1};f
    % else time=15;
    % end
    %
    % Fc=zeros(size(F));
    % for j=1:size(F,2)
    % junk=F(:,j);
    % %         junk=junk;
    % numframes=length(junk);
    % % numwindow=numframes/(Fs*15);
    % % % numwindows=50;
    % %  window=round(numframes/numwindow);
    % window=round(Fs*time);
    %         junk2=zeros(size(junk));
    %         for k=1:length(junk)
    %             cut=junk(max(1,k-window):min(numframes,k+window));
    %             cutsort=sort(cut);
    %             a=round(length(cut)*.08);
    %             junk2(k)=cutsort(a);
    %         end
    %         Fc(:,j)=(junk./junk2);
    %         maxval=max(Fc(:,j));
    %         Fc(:,j)=(Fc(:,j)-median(Fc(:,j)))/max((Fc(:,j)-median(Fc(:,j))));
    %         Fc(:,j)=maxval*Fc(:,j);
    % end