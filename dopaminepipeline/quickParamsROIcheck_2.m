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
mouse_id2=[156]


days = uipickfiles;
% for kk=1
%     mouse_id=mouse_id2(kk);
%     days=[9:22];
%     days_all=[18:31];

    for day=1:length(days)
        for plane=1:4
%             dir_1=strcat('F:\D',num2str(days_all(day)),'_RRauto',num2str(days(day)),'\suite2p\plane',num2str(plane-1),'\reg_tif');
dir_1=strcat(days{day},'\suite2p\plane',num2str(plane-1),'\reg_tif')
            cd(dir_1)

            load('params')
            find_figure(strcat('meanimg',num2str(mouse_id2),num2str(day))); hold on

            subplot(2,2,plane),imagesc(params.mimg); hold on
            colormap("gray")
            for jj = 1:length(params.newroicoords)
                if jj == 1
                    subplot(2,2,plane),plot(params.newroicoords{jj}(:,1),params.newroicoords{jj}(:,2),'y','LineWidth',1.5)
                else
                    subplot(2,2,plane), plot(params.newroicoords{jj}(:,1),params.newroicoords{jj}(:,2),'g','LineWidth',1.5)
                end
            end
        end
    end
% end