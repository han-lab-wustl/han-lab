function [dFF, F0] = redo_dFF_from_cellpose(fmatfl,Fs,time)
load(fmatfl);
Fmat = load(fmatfl); % to avoid conflicts with iscell function
F = F(logical(Fmat.iscell(:,1)),:);
Fneu = Fneu(logical(Fmat.iscell(:,1)),:);
dFF=zeros(size(F));
Fc=zeros(size(F));
% time=300; %size of moving avg window (s)
% Fs = 31.25
window=round(Fs*time);
numframes = size(F,2);
% Fs=zeros(size(F));
FminusN=bsxfun(@minus,F,.82*Fneu);
for j=1:size(F,1)
    junk=FminusN(j,:);
    
    junk2=zeros(size(junk));
    for k=1:length(junk)
        cut=junk(max(1,k-window):min(numframes,k+window));
        cutsort=sort(cut);
        a=round(length(cut)*.08);
        junk2(k)=cutsort(a);
    end
    Fc(j,:)=(junk./junk2);
    maxval=max(Fc(j,:));
    Fc(j,:)=(Fc(j,:)-1)/max((Fc(j,:)-1));
    Fc(j,:)=maxval*Fc(j,:);
    dFF(j,:)=(junk-junk2)./junk2;
    F0=mean(junk2);
    %Fc(:,i)=(junk-junk2)-mean((junk-junk2));
end
dFF_iscell = dFF;
save(fmatfl, 'dFF_iscell', '-append')
end