Fs = 31.25;
time = 300; % ms
Fmat  = dir(fullfile('G:\E135\Day1', '**', 'plane*', 'Fall.mat'));

for i=1:length(Fmat)
    disp(Fmat(i).folder)
    [dff,f0] = redo_dFF_from_cellpose(fullfile(Fmat(i).folder, Fmat(i).name), ...
        Fs, time);

end