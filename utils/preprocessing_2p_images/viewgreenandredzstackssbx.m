
 [filename,filepath]=uigetfile('*.sbx','Choose SBX file');
 dir='uni'; 

% win=60*Fs;

cd (filepath); %set path
stripped_filename=regexprep(filename,'.sbx','');
z = sbxread(stripped_filename,1,1);
global info;

numframes = info.max_idx+1;

chtemp=sbxread(stripped_filename,0,numframes);
chtemp=double(squeeze(chtemp));
grnmat = squeeze(chtemp(1,:,:,:));
meanchannel1 = squeeze(nanmean(reshape(grnmat,size(grnmat,1),size(grnmat,2),40,size(grnmat,3)/40),3));
h = implay(mat2gray(meanchannel1));
set(h.Parent,'Name','Green Channel Z stack')
redmat = squeeze(chtemp(2,:,:,:));
meanchanel2 = squeeze(nanmean(reshape(redmat,size(redmat,1),size(redmat,2),40,size(redmat,3)/40),3));
h = implay(mat2gray(meanchanel2));
set(h.Parent,'Name','Red Channel Z stack')