% celltrack category figure
% figure 2 c
clear all
cat = [39 5 11 0; 28 20 7 0; 42 3 10 0; 38 3 11 3; 41 0 13 1; 40 0 6 2;...
    66 0 4 4; 7 0 10 0 ; 31 0 6 0]; % days e186 and e201
figure('Renderer','painters')
bar(mean(cat,1,'omitnan'), 'FaceColor', 'w'); hold on
x = [repelem(1, 9) repelem(2, 9) repelem(3, 9) repelem(4, 9)];
y = cat(:);
swarmchart(x,y,'ko')
for i=1:size(cat,1)
    plot([1:4], cat(i,:), 'k')
end
ylabel('Un-Tracked Cells')
xticklabels({"Inactive/no ROI", "Obscured FOV/Surrounding Cells", "Active/no ROI", "Missing"})
xtickangle(45)
box off
