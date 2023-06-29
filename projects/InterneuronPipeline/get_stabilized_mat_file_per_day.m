
function get_stabilized_mat_file_per_day
clear all

nplanes = input('No. of planes? ');

for i = 1:nplanes
    [path{i}] = uigetdir('',['Select the folder with stabilized recordings for plane ',...
         num2str(i)]);
end

for i = 1:nplanes
    files = dir([path{i},'\*.tif']);
    video = [];
    disp(['Stitching all tiff files for plane ',num2str(i)])
    for ii = 1 : size(files, 1)
        filename = [path{i},'\',files(ii).name];
        temp = read_all_tiff_frames(filename);
        video = cat(3,video,temp);
    end
    name = files(1).name(1:(end-6));
    disp(['Saving mat file for plane ',num2str(i)])
    save([path{i},'\',name,'_XC_','plane_',num2str(i),'.mat'],'video', '-v7.3');
end