<<<<<<< HEAD

% This forms 1 big matrix with registered cells numbers across all days 

function multiple_days_reg_file_cont

clear all 
nplane = input('Which Plane are you editing? ');
[allfile,allpath] = uigetfile('*_all_days.mat','Select the register file you had previously made.');
prevreg = load([allpath allfile])
iscell = prevreg.iscell;
currentsize = size(iscell,2);
numadd = input(['You currently have ', num2str(currentsize), ' days (including the template day). How many would you like to add?']);
for i = 1:numadd
    position(i) = 10000;
    while position(i)>currentsize
    position(i) = input(['Where would you like to add new day ' num2str(i), '? (1 = position 1, 2 = between old days 1 and 2, 3 = between old days 2 and 3, max = ', num2str(currentsize),' : ']);
    end
end
[dummysort,Isort] = sort(position);
newiscell = iscell;
for i = 1:numadd
    if dummysort(i) > 1
    newiscell = cat(2,newiscell(:,1:dummysort(i)-1),zeros(size(iscell,1),1),newiscell(:,dummysort(i):end));
    
    else
    newiscell = cat(2,zeros(size(iscell,1),1),newiscell);
    end
    if i == 1
    else
    dummysort = dummysort+1;
    position(Isort(i)) = dummysort(i);
    end
end
for i = 1:numadd
        [file{i},path{i}] = uigetfile('*reg.mat',['Select the registered file for plane ',...
            num2str(nplane),' on new day ',num2str(position(i)),'and make sure it is not the template day']);
end

for i = 1:numadd
    reg_file(i) = load([path{i} file{i}]);
end 

template_intersections = newiscell(:,end);

for i = 1:numadd
     temp = intersect(template_intersections,reg_file(i).regi.rois.iscell_idcs(:,2));
     clear template_intersections
     template_intersections = temp;
     clear temp
end 
 if length(template_intersections) < size(newiscell,1)
     [~,deleteidx] = setxor(newiscell(:,end),template_intersections);
      newiscell(deleteidx,:) = [];
 end

for i = 1:numadd
    day_cells = reg_file(i).regi.rois.iscell_idcs(:,1);
    temp_cells = reg_file(i).regi.rois.iscell_idcs(:,2);
    for j = 1:length(template_intersections)
    idx = find(temp_cells == template_intersections(j));
    newiscell(j,position(i)) = day_cells(idx);
    end  
    
    clear day_cells temp_cells idx
end 

clearvars iscell

iscell = newiscell;

saveName = input('Enter the new saving name (Mnumber_D1_D2_D3...DTemplatenumber, use quotes):');
saveName = [saveName '_plane ' num2str(nplane) '_all_days.mat'];

save([allpath saveName],'iscell');
end 

=======

% This forms 1 big matrix with registered cells numbers across all days 

function multiple_days_reg_file_cont

clear all 
nplane = input('Which Plane are you editing? ');
[allfile,allpath] = uigetfile('*_all_days.mat','Select the register file you had previously made.');
prevreg = load([allpath allfile])
iscell = prevreg.iscell;
currentsize = size(iscell,2);
numadd = input(['You currently have ', num2str(currentsize), ' days (including the template day). How many would you like to add?']);
for i = 1:numadd
    position(i) = 10000;
    while position(i)>currentsize
    position(i) = input(['Where would you like to add new day ' num2str(i), '? (1 = position 1, 2 = between old days 1 and 2, 3 = between old days 2 and 3, max = ', num2str(currentsize),' : ']);
    end
end
[dummysort,Isort] = sort(position);
newiscell = iscell;
for i = 1:numadd
    if dummysort(i) > 1
    newiscell = cat(2,newiscell(:,1:dummysort(i)-1),zeros(size(iscell,1),1),newiscell(:,dummysort(i):end));
    
    else
    newiscell = cat(2,zeros(size(iscell,1),1),newiscell);
    end
    if i == 1
    else
    dummysort = dummysort+1;
    position(Isort(i)) = dummysort(i);
    end
end
for i = 1:numadd
        [file{i},path{i}] = uigetfile('*reg.mat',['Select the registered file for plane ',...
            num2str(nplane),' on new day ',num2str(position(i)),'and make sure it is not the template day']);
end

for i = 1:numadd
    reg_file(i) = load([path{i} file{i}]);
end 

template_intersections = newiscell(:,end);

for i = 1:numadd
     temp = intersect(template_intersections,reg_file(i).regi.rois.iscell_idcs(:,2));
     clear template_intersections
     template_intersections = temp;
     clear temp
end 
 if length(template_intersections) < size(newiscell,1)
     [~,deleteidx] = setxor(newiscell(:,end),template_intersections);
      newiscell(deleteidx,:) = [];
 end

for i = 1:numadd
    day_cells = reg_file(i).regi.rois.iscell_idcs(:,1);
    temp_cells = reg_file(i).regi.rois.iscell_idcs(:,2);
    for j = 1:length(template_intersections)
    idx = find(temp_cells == template_intersections(j));
    newiscell(j,position(i)) = day_cells(idx);
    end  
    
    clear day_cells temp_cells idx
end 

clearvars iscell

iscell = newiscell;

saveName = input('Enter the new saving name (Mnumber_D1_D2_D3...DTemplatenumber, use quotes):');
saveName = [saveName '_plane ' num2str(nplane) '_all_days.mat'];

save([allpath saveName],'iscell');
end 

>>>>>>> 754f532e47d152334ffae033cf3e5763ab9bf2c0
