function [Fallout,forwardvelout,cpp,yout,rewout,F0out,masks,mimg,Fallnb,MouseID]=...
    GMNeuronStructor(varargin)
%% [Fallout,rotationout,forwardvelout,cpp,yout,rewout,F0out,masks,mimg,Fallnb,MouseID]=...
%   NeuronStructor(varargin)
% () - manually select Fs/Planes/ID/ etc
% (Frame Rate, Num Planes, ID, File Names, Path Names, Remap (0,1), age
% (new/old), etl settings, threshold
% saveDir='G:\MA Data\Interneurons\';
if nargin<1
    Fs=input('Fs? ');
    planes=input('Number of planes: ');
    MouseID = input('Mouse ID and Day: ');
    manual=1;
else
    Fs=varargin{1};
    planes=varargin{2};
    MouseID=varargin{3};
    manual=0;
end
if nargin>5
    remap=varargin{6};
else remap=0;
end
if nargin>6
    age=varargin{7};
end
if nargin>7
    etl=varargin{8};
end
if nargin>8
    cell_thresh=varargin{9};
else cell_thresh=.1;
end
if nargin>9
    curated=varargin{10};
end

paths{planes}=0;
names{planes}=0;
for p=1:planes
    if manual==1
        disp(['Plane ', num2str(p)]);
        [names{p},paths{p}]=uigetfile('*.mat','pick your files');
    else
        names{p}=varargin{4}{p};
        paths{p}=varargin{5}{p};
    end
end
load([paths{1},names{1}]);
mimg{planes}=[];
if exist('frame','var')
    for p=1:planes
        load([paths{p},names{p}]);
        mimg{p}=frame;
    end
elseif  exist('meanImage','var')
    for p=1:planes
        load([paths{p},names{p}]);
        mimg{p}=meanImage;
    end
end
rewL=rewards;
[F,ybin,forward,rotation,cpp,masks,Fraw,rew,nF]=align_running(paths,names,planes);
if size(F,2)>60
F=redo_dFF(Fraw,Fs,30,nF);    
else
% F=redo_dFF(Fraw,Fs,900,nF);
fs=sort(Fraw);
f8=fs(round(length(fs)*.08),:);
F=bsxfun(@rdivide,Fraw,f8)-1;
end
Fnb=redo_dFF(Fraw,Fs,15,nF);
% To test statistic to identify good cells
% keep=check_cells(F,saveDir,(MouseID),.3,Fs);
% F(:,~keep)=nan;
% Fraw(:,~keep)=nan;

% if ~exist('curated','var')
%     keep=check_cells(F,saveDir,(MouseID),cell_thresh);
%     F(:,~keep)=nan;
%     Fraw(:,~keep)=nan;
% else
%     F=F(:,curated);
%     Fraw=Fraw(:,curated);
% end

if remap
    Fall{4}=[];
    Frawall=Fall;
    Fallnb=[];    
    forwardvelcell{4}=[];
    rotationvelcell{4}=[];
    ybinnedcell{4}=[];
    times{4}=[];
    pos{4}=[];
    rewardscell{3}=[];
    rewratio=length(rewL)/(length(F));
    env_label={' Familiar',' Novel',' Familiar2','All'};
    novel_start=round(timeSplit(1)/rewratio);
    novel_end=round(timeSplit(2)/rewratio);
    rewinds{1}=1:timeSplit(1);
    rewinds{2}=(timeSplit(1)+1):timeSplit(2);
    rewinds{3}=(timeSplit(2)+1):length(rewL);
    rewinds{4}=1:length(rewL);
    envinds{1}=1:novel_start;
    envinds{2}=(novel_start+1):novel_end;
    envinds{3}=(novel_end+1):length(F);
    envinds{4}=1:length(F);
    for env=1:length(env_label)
        rewardscell{env}=rew(round(envinds{env}));
        forwardvelcell{env}=forward(envinds{env},:);
        rotationvelcell{env}=rotation(envinds{env},:);
        ybinnedcell{env}=ybin(envinds{env},:);
        temp=bsxfun(@minus,ybinnedcell{env},min(ybinnedcell{env},[],1));
        pos{env}=ceil(temp/max(temp(:))*180/5+eps);
        times{env}=linspace(1,length(ybinnedcell{env})/Fs,length(ybinnedcell{env}));
        Fall{env}=F(envinds{env},:);
        Fallnb{env}=Fnb(envinds{env},:);
        Frawall{env}=Fraw(envinds{env},:);
        if ~isempty(nF)
        nFall{env}=nF(envinds{env},:);
        end
    end
else
    ybinnedcell{1}=ybin;
    temp=bsxfun(@minus,ybin,min(ybin,[],1));
    pos{1}=ceil(temp/max(temp(:))*180/5+eps);
    rotationvelcell{1}=rotation;
    forwardvelcell{1}=forward;
    env_label={''};
    Fall{1}=F;
    Fallnb{1}=Fnb;
    Frawall{1}=Fraw;
    spk{1} = spk;
   if ~isempty(nF)
    nFall{1}=nF;
   end
    
    rewardscell{1}=rew;
    dists{1}=0;
    corrs{1}=0;
end

F0out.raw=Frawall;
        if ~isempty(nF)
F0out.neuro=nFall;
        end
Fallout=Fall;
rotationout=rotationvelcell;
forwardvelout=forwardvelcell;
yout=ybinnedcell;
rewout=rewardscell;
