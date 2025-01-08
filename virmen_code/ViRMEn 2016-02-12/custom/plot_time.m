% plot the time of each iteration
% open the binary file
fid = fopen('timeLog.dat');
% read all data from the file into a 5-row matrix
data = fread(fid,'double');
% close the file
fclose(fid);
% plot the 2D position information
plot(diff(data),'k.');
ylabel('ms')