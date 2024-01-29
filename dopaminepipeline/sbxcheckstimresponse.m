close all
% clear all
pr_dir0 = uipickfiles;
% [tifffilename,tiffpath]=uigetfile('*.sbx','pick your sbx file');
%tic
clearvars -except pr_dir0

for days = 1:length(pr_dir0)
    tiffpath = pr_dir0{days};
cd (tiffpath); %set path
dumb = dir;
for lissst = 1:length(dumb)
temp = dumb(lissst).name;
    if contains(temp,'sbx')
        tifffilename = temp;
    end
end

stripped_tifffilename=regexprep(tifffilename,'.sbx','');
z = sbxread(stripped_tifffilename,1,1);
 global info;
    frameload=info.max_idx+1; %to load all frames
newmov = sbxread(stripped_tifffilename,0,frameload);
newmov = squeeze(newmov);
%%
stimmean = squeeze(nanmean(nanmean(newmov(25:50,:,:))));
stimmean2 = squeeze(nanmean(nanmean(newmov(200:400,200:600,:))));
%%
numplanes = size(info.etl_table,1);
stims = zeros(1,length(stimmean)); 
for allplanes =1:numplanes
    temp = allplanes:numplanes:length(stimmean);
    temp2 = find(abs(diff(stimmean(temp)))>200)+1;
%     temp2(find(abs(diff(temp2))<3)) = [];
    stims(temp(temp2)) = 1;
end
stims(1:10) = 0;
        temp3 = find(stims);
%         stims(temp3(find(diff(temp3)<200)+1)) = 0;

figure;
for pl = 1:numplanes
    subplot(numplanes,1,numplanes-pl+1)
    plot(pl:numplanes:length(stimmean),stimmean2(pl:numplanes:end))
    hold on; 
    ylims = ylim;
    plot(rescale(stims,ylims(1),ylims(2)))
end
title(['Day ' num2str(days)])
end