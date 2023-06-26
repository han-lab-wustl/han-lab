%% align_running  - fixes sampling differences between Fcs and abf
% input (pnum) or ({paths},{names},{pnum}
%output:
%[dFF,ybinned,forward,rotation,cell_per_plane,masks,Fraw,rewards,neuropilF,spks(if
%relevant, licks, envInds]
%GM edits for HRZ------
 %removed rotation. added time, rewloc per epoch, lick variable, and trial
 %number 

function [Fall,y2,f2,cell_per_plane,mask,Fraw,rewsmall2,nFall,timeout,RewLocout,lickout,trialnumout,spksout]=...
    HRZalign_running(varargin)
if nargin==1
    pnum=varargin{1};
    for p=1:pnum
        [names{p},paths{p}]=uigetfile('*.mat');
    end
else
    paths=varargin{1};
    names=varargin{2};
    pnum=varargin{3};
end
load([paths{1},names{1}]);

if exist('Fca','var')
    dFF=Fca;
elseif ~exist('dFF','var')
    dFF=Fc;
end
if ~exist('masks','var')
    masks=0;
end
if ~exist('F','var')
    F=Fca;
end
if size(dFF,2)>size(Fc,2)
   rewrite_dFF_fix_mismatch(names,paths); 
end
lickout=[];
Fall=[];
Fraw=[];
spksout=[];
nFall=[];
y2 = [];
f2 = [];
timeout = [];
RewLocout = [];
lickout = [];
trialnumout = [];
rewsmall2 = [];
if pnum==1
    Fall=dFF;
    Fraw=F;
    y2=repmat(ybinned,1,size(dFF,2));
    f2=repmat(forwardvel,1,size(dFF,2));
    timeout = repmat(timedFF,1,size(dFF,2));
    RewLocout =repmat(changeRewLoc,1,size(dFF,2));
    lickout = repmat(licks,1,size(dFF,2));
    trialnumout = repmat(trialnum,1,size(dFF,2));
    rewsmall2 = repmat(lgocial(rewards),1,size(dFF,2));
    cell_per_plane=size(dFF,2);
    mask{1}=masks;
    if exist('spks','var')
        spksout=spks;
    end
    if exist('nF','var')
       nFall=nF; 
    else
        nF=zeros(size(Fall));
       nFall=nF; 
    end
    
end
num_cells=size(dFF,2);
ysave=ybinned;
fsave=forwardvel;

if pnum>1
    cell_per_plane(pnum)=0;
    mask{pnum}=0;
    for p=1:pnum
        %         clear Fc2 Fc Fca
        %         if exist('Fca','var')
        if ~isempty(names{p})
        load([paths{p},names{p}]);
        end
        %         else
        %             load([paths{p},names{p}],'F','Fc2','Fc','masks');
        %         end
        if exist('Fca','var')
            dFF=Fca;    F=Fca;
        end
        if ~exist('masks','var')
            masks=0;
        end
        
        mask{p}=masks;
        if exist('Fc','var')
            cell_per_plane(p)=size(dFF,2);
            if ~isempty(Fall) && ~isempty(dFF)
                if length(Fall)<length(dFF)
                    Fall=[Fall,Fc(1:length(Fall),:)];
%                     Fall=[Fall,dFF(1:length(Fall),:)];
                    Fraw=[Fraw,F(1:length(Fall),:)];
                    nFall=[nFall,nF(1:length(Fall),:)];
                else
                    Fall=[Fall(1:length(dFF),:), Fc];
%                     Fall=[Fall(1:length(dFF),:), dFF];
                    Fraw=[Fraw(1:length(dFF),:), F];
                    nFall=[nFall(1:length(dFF),:), nF];
                end
                if exist('spks','var')
                    spksout=[spksout, spks];
                end
            else
                Fall=Fc;
%                 Fall=dFF;
                Fraw=F;
                nFall=nF;
                if exist('spks','var')
                    spksout=spks;
                end
            end
        else
            cell_per_plane(p)=0;
        end
        clear Fc2 Fc
        y2=[y2 repmat(ybinned',1,cell_per_plane(p))];
        f2= [f2 repmat(forwardvel',1,cell_per_plane(p))];
        timeout= [timeout repmat(timedFF',1,cell_per_plane(p))];
        RewLocout= [RewLocout repmat(changeRewLoc',1,cell_per_plane(p))];
        lickout= [lickout repmat(logical(licks)',1,cell_per_plane(p))];
        trialnumout= [trialnumout repmat(trialnum',1,cell_per_plane(p))];
        rewsmall2= [rewsmall2 repmat(rewards',1,cell_per_plane(p))];
    end
    
    
end
if exist('timeSplit','var') && ~isempty(timeSplit)
    tsDS=timeSplit/(length(rewards)/length(Fall));
   envInds{1}=1:round(tsDS(1));
   envInds{2}=round(tsDS(1)+1):round(tsDS(2));
   envInds{3}=(round(tsDS(2))+1):length(Fall);
   envInds{4}=1:length(Fall);
   
end

end
function for1=fix_trace(forwardvel)
for1=forwardvel;
d=[0;diff(forwardvel)];
dd=[0;diff(d)];
for1(abs(dd)<.05)=NaN;
for2=for1;

for2(isnan(for2))=[];
for2=smooth(for2,9);
d2=[0;diff(for2)];
for2(abs(d2)>.03)=NaN;
for3=for2;

for3(isnan(for3))=[];
d3=[0;diff(for3)];
for3(abs(d3)>.03)=NaN;
for4=for3;

for4(isnan(for4))=[];
for4=smooth(for4);
d4=[0;diff(for4)];
for5=for4;
for5(abs(d4)>.03)=NaN;
tidx=1:length(for4);
for5=real(interp1(tidx(~isnan(for5)),for5(~isnan(for5)).^11,tidx,'pchip','extrap').^(1/11))';
for3(~isnan(for3))=for5;
tidx=1:length(for3);
for3=real(interp1(tidx(~isnan(for3)),for3(~isnan(for3)).^11,tidx,'pchip','extrap').^(1/11))';
for2(~isnan(for2))=for3;
tidx=1:length(for2);
for2=real(interp1(tidx(~isnan(for2)),for2(~isnan(for2)).^11,tidx,'pchip','extrap').^(1/11))';
for1(~isnan(for1))=for2;
tidx=1:length(for1);
for1=real(interp1(tidx(~isnan(for1)),for1(~isnan(for1)).^11,tidx,'pchip','extrap').^(1/11))';
end