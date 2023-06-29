<<<<<<< HEAD

% This forms 1 big matrix with registered cells numbers across all days 

function multiple_days_reg_file_GM

clear all 
ndays = input('No. of days you want to register-including template day'); 
nplane = input('Plane that you want to register');

for i = 1:ndays-1
        [file{i},path{i}] = uigetfile('*.mat',['Select the registered file for ',...
            num2str(nplane),' plane on day ',num2str(i),'and make sure it is not the template day']);
        cd(path{i})
        i
end

for i = 1:ndays-1
    reg_file(i) = load([path{i} file{i}]);
end 

template_intersections = reg_file(1).regi.rois.iscell_idcs(:,2);
if ~isempty(template_intersections > 0)

for i = 1:ndays-1
     temp = intersect(template_intersections,reg_file(i).regi.rois.iscell_idcs(:,2));
     clear template_intersections
     template_intersections = temp;
     clear temp
end 

all_days_cell = zeros(length(template_intersections),ndays);

all_days_cell(:,ndays) = template_intersections;
for i = 1:ndays-1
    day_cells = reg_file(i).regi.rois.iscell_idcs(:,1);
    temp_cells = reg_file(i).regi.rois.iscell_idcs(:,2);
    for j = 1:length(template_intersections)
    idx = find(temp_cells == template_intersections(j));
    all_days_cell(j,i) = day_cells(idx);
    end  
    
    clear day_cells temp_cells idx
end 

iscell = all_days_cell;

uisave('iscell');
end 

=======

% This forms 1 big matrix with registered cells numbers across all days 

function multiple_days_reg_file_GM

clear all 
ndays = input('No. of days you want to register-including template day'); 
nplane = input('Plane that you want to register');

for i = 1:ndays-1
        [file{i},path{i}] = uigetfile('*.mat',['Select the registered file for ',...
            num2str(nplane),' plane on day ',num2str(i),'and make sure it is not the template day']);
        cd(path{i})
        i
end

for i = 1:ndays-1
    reg_file(i) = load([path{i} file{i}]);
end 

template_intersections = reg_file(1).regi.rois.iscell_idcs(:,2);
if ~isempty(template_intersections > 0)

for i = 1:ndays-1
     temp = intersect(template_intersections,reg_file(i).regi.rois.iscell_idcs(:,2));
     clear template_intersections
     template_intersections = temp;
     clear temp
end 

all_days_cell = zeros(length(template_intersections),ndays);

all_days_cell(:,ndays) = template_intersections;
for i = 1:ndays-1
    day_cells = reg_file(i).regi.rois.iscell_idcs(:,1);
    temp_cells = reg_file(i).regi.rois.iscell_idcs(:,2);
    for j = 1:length(template_intersections)
    idx = find(temp_cells == template_intersections(j));
    all_days_cell(j,i) = day_cells(idx);
    end  
    
    clear day_cells temp_cells idx
end 

iscell = all_days_cell;

uisave('iscell');
end 

>>>>>>> 754f532e47d152334ffae033cf3e5763ab9bf2c0
