
function tiff_stack = read_all_tiff_frames(filename)
%%
% filename should be in .tif format
%%
tiff_info = imfinfo(filename); % return tiff structure, one element per image
tiff_stack = imread(filename, 1) ;
for ii = 2 : size(tiff_info, 1)
    ii
    temp_tiff = imread(filename, ii);
    tiff_stack = cat(3 , tiff_stack, temp_tiff);
end