function [peridffs_s]= plot_perireward_per_plane(mouse_name, dy, pln,fmatfl, rnng, dst)
% makes a 2x2 figure with 
% 1) dff aligned to speed and behavior
% 2) peri single rewards
% 3) peri double rewards
% 4) peri solenoid rewards
% based on the Pavlovian 'random rewards' task and after running batch and vrdarkrewards script 
% from munni's pipeline (see 'projects/dopamine')
% TODO: incorporate perirewardbinnedactivity into this
plane=load(fullfile(fmatfl(pln).folder, fmatfl(pln).name));
figure;
subplot(2,2,1);
% based on munni's code
% use roibasemean3 because this is what is used for just the drawn ROIs 
%         smooth_mean = smoothdata(plane.params.roibasemean3{1}','gaussian',5); % gaussian window = 5
smooth_mean = plane.params.roibasemean3{1}';
plot(smooth_mean, 'g'); hold on
yyaxis right
plot(plane.forwardvel/2, 'k-')
plot(plane.rewards*100, 'r-')
plot(plane.solenoid2*100, 'b-')
legend({'smooth base mean dff', 'velocity', 'rewards', 'solenoid'})
title(sprintf('%s, day %i, plane %i', mouse_name, dy, pln));
% peri reward        
subplot(2,2,2);
rewidx = find(plane.rewards==1);
rewidx = rewidx(rewidx>1000); % exclude early rewards
peridffs_s = ones(length(rewidx), rnng*2)*NaN; % to save
for rewid=1:length(rewidx)
    rng = rewidx(rewid)-rnng:rewidx(rewid)+rnng-1;
    if ~sum(rng<=0)>0 && ~sum(rng>length(smooth_mean))>0 % super early rewards or 
        % rewards too close to end of imaging are excluded                
        peridff = smooth_mean(rng);
        plot(peridff); hold on            
        peridffs_s(rewid,:)=peridff;
    end
end
plot(mean(peridffs_s,'omitnan'),'k','LineWidth',2);         
xline(median(1:rnng*2),'-.b','Reward'); %{'Conditioned', 'stimulus'}
title({sprintf('%s, day %i, plane %i', mouse_name, dy, pln)}, {'Peri single reward'});
% peri reward        
subplot(2,2,3);
rewidx = find(plane.rewards==2);
peridffs = ones(length(rewidx), rnng*2)*NaN;
for rewid=1:length(rewidx)
    rng = rewidx(rewid)-rnng:rewidx(rewid)+rnng-1;
    if ~sum(rng<=0)>0 && ~sum(rng>length(smooth_mean))>0 % super early rewards or 
        % rewards too close to end of imaging are excluded                
        peridff = smooth_mean(rng);
        plot(peridff); hold on            
        peridffs(rewid,:)=peridff;
    end
end
plot(mean(peridffs,'omitnan'),'k','LineWidth',2);         
xline(median(1:rnng*2),'-.b','Reward'); %{'Conditioned', 'stimulus'}
title({sprintf('%s, day %i, plane %i', mouse_name, dy, pln)}, {'Peri double reward'});
% peri solenoid        
subplot(2,2,4);
rewidx = find(plane.solenoid2==1);
peridffs = ones(length(rewidx), rnng*2)*NaN;
for rewid=1:length(rewidx)
    rng = rewidx(rewid)-rnng:rewidx(rewid)+rnng-1;
    if ~sum(rng<=0)>0 && ~sum(rng>length(smooth_mean))>0 % super early rewards or 
        % rewards too close to end of imaging are excluded                
        peridff = smooth_mean(rng);
        plot(peridff); hold on            
        peridffs(rewid,:)=peridff;
    end
end
plot(mean(peridffs,'omitnan'),'k','LineWidth',2);         
xline(median(1:rnng*2),'-.b','CS'); %{'Conditioned', 'stimulus'}
title({sprintf('%s, day %i, plane %i', mouse_name, dy, pln)}, {'Peri solenoid2'});
savefig(fullfile(dst, sprintf('%s_day%03d_plane%d.fig', mouse_name, dy, pln)))
end