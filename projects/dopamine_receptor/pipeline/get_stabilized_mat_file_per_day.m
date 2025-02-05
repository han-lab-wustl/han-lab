function get_stabilized_mat_file_per_day(nplanes, planeflds)

for i = 1:nplanes
    pth = fullfile(planeflds(i).folder, planeflds(i).name, 'reg_tif');
    files = dir([pth,'\*.tif']);
    video = [];
    disp(['Stitching all tiff files for plane ',num2str(i)])
    for ii = 1:size(files, 1)        
        filename = [pth,'\',files(ii).name];
        temp = read_all_tiff_frames(filename);
        video = cat(3,video,temp);
    end
    name = files(1).name(1:(end-6));
    disp(['Saving mat file for plane ',num2str(i)])
    save([pth,'\',name,'_XC_','plane_',num2str(i),'.mat'],'video', '-v7.3');
end
end
