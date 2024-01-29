paramfile = uipickfiles;


load(paramfile{1})

figure;
imagesc(params.mimg)
colormap("gray")
hold on
for jj = 1:length(params.newroicoords)
    if jj == 1
        plot(params.newroicoords{jj}(:,1),params.newroicoords{jj}(:,2),'y','LineWidth',1.5)
    else
        plot(params.newroicoords{jj}(:,1),params.newroicoords{jj}(:,2),'g','LineWidth',1.5)
    end
end
%%
mouse_id2=[167:171]
for kk=1:5
    mouse_id=mouse_id2(kk);
    day=13
    for plane=1:4
        dir_1=strcat( 'I:\HRZ_Batch2\E',num2str(mouse_id),'\Day_',num2str(day),'\suite2p\plane',num2str(plane-1),'\reg_tif')
        %     dir_1=strcat('G:\dark_reward\E',num2str(mouse_id),'\Day_',num2str(day),'\suite2p\plane',num2str(plane-1),'\reg_tif');
        cd(dir_1)
        load('params')
        find_figure(strcat('meanimg',num2str(mouse_id))); hold on
        
        subplot(2,2,plane),imagesc(params.mimg)
        colormap("gray")
        hold on
        for jj = 1:length(params.newroicoords)
            if jj == 1
                subplot(2,2,plane),plot(params.newroicoords{jj}(:,1),params.newroicoords{jj}(:,2),'y','LineWidth',1.5)
            else
                subplot(2,2,plane), plot(params.newroicoords{jj}(:,1),params.newroicoords{jj}(:,2),'g','LineWidth',1.5)
            end
        end
        
    end
end