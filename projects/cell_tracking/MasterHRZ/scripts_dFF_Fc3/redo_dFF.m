function [dFF,Fc]=redo_dFF(F,Fs,varargin)
if nargin>2
    time=varargin{1};
else time=300;
end
if nargin>3
    nF=varargin{2};
else nF=[];
end
if nargin>4
    plotOpt =1;
    figure
else plotOpt=0;
end
    
% F=mouse(3).F0{14}.raw{4};
disp ('calculating dFF')
tic
dFF=zeros(size(F));
for j=1:size(F,2)
    if ~isempty(nF)
%     junk=F(:,j)-.6*nF(:,j);
%     if mean(junk)<0
        junk=F(:,j)-nF(:,j)+mean(nF(:,j));
%     end
    else
        junk=F(:,j);
    end
numframes=length(junk);
window=round(Fs*time);

        junk2=zeros(size(junk));
        for k=1:length(junk)
            cut=junk(max(1,k-window):min(numframes,k+window));
            cutsort=sort(cut);
            a=round(length(cut)*.08);
            junk2(k)=cutsort(a);
        end
        x=1:numframes;
        if ~isnan(mean(junk))
        fitData=fit((x)',junk2,'exp2');
%         xBase=((fitData.a)*exp((fitData.b)*(x))+(fitData.c)*exp((fitData.d)*(x)))';
        xBase=junk2;
        
        dFF(:,j)=((junk)-(xBase))./xBase;
        Fc(:,j)=junk./junk2;
            maxval=max(Fc(:,j));
    Fc(:,j)=(Fc(:,j)-1)/max((Fc(:,j)-1));
    Fc(:,j)=maxval*Fc(:,j);
               
        
        else
            dFF(:,j)=nan(size(F(:,j)));
        end
        if plotOpt
            nSq=ceil(sqrt(size(F,2)));
            subplot(nSq,nSq,j)
            plot(junk);
            hold on;
            plot(xBase);
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
toc
% Old dFF method
% if nargin>2
%     time=varargin{1};f
% else time=15;
% end
% 
% dFF=zeros(size(F));
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
%         dFF(:,j)=(junk./junk2);
%         maxval=max(dFF(:,j));
%         dFF(:,j)=(dFF(:,j)-median(dFF(:,j)))/max((dFF(:,j)-median(dFF(:,j))));
%         dFF(:,j)=maxval*dFF(:,j);
% end