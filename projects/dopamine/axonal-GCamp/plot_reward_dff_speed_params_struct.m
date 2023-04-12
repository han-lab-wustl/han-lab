% Zahra
% axonal-dopamine analysis
% after running batch and vrdarkrewards script (but not using any of the
% figures...)

mouse_name = "e193";
days = [10:17];
src = "X:\dopamine_imaging";
planes = 3;
for dy=days    
    fmatfl = dir(fullfile(src, mouse_name, string(dy), '**\params.mat'));     % finds all params files
    for pln=1:planes
        plane=load(fullfile(fmatfl(pln).folder, fmatfl(pln).name));
        figure;
        % based on munni's code
        % use roibasemean3 because this is what is used for just the drawn ROIs 
        smooth_mean = smoothdata(plane.params.roibasemean3{1}','gaussian',5); % gaussian window = 5
        plot(smooth_mean, 'g'); hold on
        yyaxis right
        plot(plane.forwardvel/2, 'k-')
        plot(plane.rewards*100, 'r-')
        plot(plane.solenoid2*100, 'b-')
        figure;plot(plane.params.raw_mean)
        legend({'smooth base mean dff', 'velocity', 'rewards', 'solenoid'})
        title(sprintf('%s, day %i, plane %i', mouse_name, dy, pln));
    end
end