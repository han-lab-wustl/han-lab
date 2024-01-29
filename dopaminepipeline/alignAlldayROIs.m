%fixes all variables across days that don't have the right amount of ROIs
 clear all
close all
wrk_dir = uipickfiles('Prompt','Pick the Workspace you would like to add');

load(wrk_dir{1})
msgbox(pr_dir0(:))
day_dir = uipickfiles('Prompt','Please Select all the folders for the previous popup');

test1 = roi_dop_alldays_planes_periCS;
roikey = ~cellfun(@isempty,test1);
refday = find(sum(roikey,2)==size(test1,2),1);
checkdays = find(sum(roikey,2)<size(test1,2));

%%
refday = 20;

    dir_s2p = struct2cell(dir([day_dir{refday} '\**\suite2p']));
    planefolders = dir_s2p(:,~cellfun(@isempty,regexp(dir_s2p(1,:),'plane')));
    
    g = 0;
   
    figure;
    allrois  = cell(1,size(test1,2));
    sb = [];
    for allplanes = 1:size(planefolders,2)
       sb(allplanes) = subplot(2,2,allplanes);
         pr_dir2=strcat(planefolders{2,allplanes},'\plane',num2str(allplanes-1),'\reg_tif\')
        load([pr_dir2 'params.mat'])
        imagesc(params.mimg)
        colormap(bone)
        hold on
        for rr = 1:length(params.newroicoords)
            g = g+1;
            plot(params.newroicoords{rr}(:,1),params.newroicoords{rr}(:,2),'y','LineWidth',1.5)
            allrois{g} = polyshape(params.newroicoords{rr});
        end
        
        
    end

%%


for c = checkdays
        dir_s2p = struct2cell(dir([day_dir{c} '\**\suite2p']));
    planefolders = dir_s2p(:,~cellfun(@isempty,regexp(dir_s2p(1,:),'plane')));
end