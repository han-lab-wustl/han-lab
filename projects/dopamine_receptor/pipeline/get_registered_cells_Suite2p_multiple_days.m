

% Thisi grabs the 

function get_registered_cells_Suite2p_multiple_days(varargin)
% clear all

%% Plane to register
if nargin < 10
    plane = input('Plane that you are registering');
    nday = input(['Day that you are registering amongst all registered days e.g. if all days\n'...
        ' involved are D9 D10 D11 D12 and D13 and you have D13 as your template day and \n'...
        'you are registering cells for D10 right now then enter 2. This is considering that \n'...
        'when you ran multiple_days_reg_file command, you selected registered files in the order of \n '...
        'D9 D10 D11 and D12']);
    days_involved = input('Enter days involved - ex : _D9_D10_D11_D12_D13 \n');
    [file_1,path_1] = uigetfile('*F.mat',['Select the registering day number',num2str(nday),' -F file for plane ',num2str(plane)]);
    filename_1 = [path_1 file_1];
    
    [file_2,path_2] = uigetfile('*F.mat',['Select the template days file F for plane ',num2str(plane)]');
    filename_2 = [path_2 file_2];
    
    [file_3,path_3] = uigetfile('*all_days.mat',['Select the registered file that has all days for plane ',num2str(plane)]');
    filename_3 = [path_3 file_3];
else
    plane = varargin{1};
    nday = varargin{2};
    days_involved = varargin{3};
    filename_1 = varargin{4};
    filename_2 = varargin{5};
    filename_3 = varargin{6};
    file_1 = varargin{7};
    path_1 = varargin{8};
    file_2 = varargin{9};
    path_2 = varargin{10};
    
end
%% Files to load - registering day, registered day, reg files for the two
registered_file = load(filename_3);
reg_day = load(filename_1);
temp_day = load(filename_2);

%% Getting registered cells idx and all field names
cells_registered = registered_file.iscell;
cells_reg_day = cells_registered(:,nday);
cells_temp_day = cells_registered(:,size(cells_registered,2));
[~,idx] = sort(cells_temp_day);
cells_temp_day = cells_temp_day(idx);
% zd added
cells_temp_day_count = 1:length(cells_temp_day);
cells_reg_day = cells_reg_day(idx);
fields = fieldnames(reg_day);
fields(strcmp(fields,'notUsed')) = []; % 8/26 gm fix for crashing when trying to find notUsed


%% Registered cells info for template day

for i = 1:length(fields)
    i
    dummy = eval(['temp_day.',fields{i}]);
    if(strcmp(fields{i},'F') == 1) || (strcmp(fields{i},'nF') == 1) ...
            || (strcmp(fields{i},'Fc') == 1) || (strcmp(fields{i},'dFF') == 1)...
            || (strcmp(fields{i},'Fc2') == 1) || (strcmp(fields{i},'spks') == 1)
        if isempty(dummy)
            eval(['regcells_temp_day.',fields{i} '= dummy']);
        else
            eval(['regcells_temp_day.',fields{i} '= dummy(:,cells_temp_day)']);
        end
    end
    
    if strcmp(fields{i}, 'masks') == 1
    fieldName = fields{i};
    regcells_temp_day.(fieldName) = dummy(cells_temp_day,:,:);
    end
    
    if(strcmp(fields{i},'Fs') == 1) || (strcmp(fields{i},'meanImage') == 1) ...
            || (strcmp(fields{i},'procImage') == 1) || (strcmp(fields{i},'frame') == 1)
        eval(['regcells_temp_day.',fields{i} '= dummy']);
    end
    
    clear dummy
end

%% Registered cells info for registering day

for i = 1:length(fields)
    i
    dummy = eval(['reg_day.',fields{i}]);
    if(strcmp(fields{i},'F') == 1) || (strcmp(fields{i},'nF') == 1) ...
            || (strcmp(fields{i},'Fc') == 1) || (strcmp(fields{i},'dFF') == 1)...
            || (strcmp(fields{i},'Fc2') == 1) || (strcmp(fields{i},'spks') == 1)
        
        if isempty(dummy)
            eval(['regcells_reg_day.',fields{i} '= dummy']);
        else
            eval(['regcells_reg_day.',fields{i} '= dummy(:,cells_reg_day)']);
        end
    end
    
    if strcmp(fields{i}, 'masks') == 1
    fieldName = fields{i};
    regcells_reg_day.(fieldName) = dummy(cells_reg_day,:,:);
    end
    
    if(strcmp(fields{i},'Fs') == 1) || (strcmp(fields{i},'meanImage') == 1) ...
            || (strcmp(fields{i},'procImage') == 1) || (strcmp(fields{i},'frame') == 1)
        eval(['regcells_reg_day.',fields{i} '= dummy']);
    end
    
    clear dummy
end

%% Showing same cells masks accordingly numbered on both days

temp_day_masks = regcells_temp_day.masks;
for i = 1:size(temp_day_masks,1)
    temp_day_cents(i) = regionprops(squeeze(temp_day_masks(i,:,:)),'centroid');
end

if exist ('temp_day_cents','Var')
    temp_day_cents = squeeze(cell2mat(struct2cell(temp_day_cents)));
    temp_day_masks = squeeze(sum(temp_day_masks,1));
    
    
    reg_day_masks = regcells_reg_day.masks;
    for i = 1:size(reg_day_masks,1)
        reg_day_cents(i) = regionprops(squeeze(reg_day_masks(i,:,:)),'centroid');
    end
    reg_day_cents = squeeze(cell2mat(struct2cell(reg_day_cents)));
    reg_day_masks = squeeze(sum(reg_day_masks,1));
    
    figure;
    hold on;
    subplot(2,2,1)
    imagesc(temp_day_masks);
    hold on
    if size(cells_registered,1) > 1 || ~isempty(cells_registered) == 0 
    for i = 1:size(cells_registered,1)
        text(temp_day_cents(1,i),temp_day_cents(2,i),num2str(i),'HorizontalAlignment','center');
    end
    else 
        for i = 1:size(cells_registered,1)
        text(temp_day_cents(i,1),temp_day_cents(i,2),num2str(i),'HorizontalAlignment','center');
        end
    end 
        
    title('Template day')
    
    subplot(2,2,3)
    imagesc(reg_day_masks);
    hold on
    if size(cells_registered,1) > 1 || ~isempty(cells_registered) == 0 
    for i = 1:size(cells_registered,1)
        text(reg_day_cents(1,i),reg_day_cents(2,i),num2str(i),'HorizontalAlignment','center');
    end
    else 
        for i = 1:size(cells_registered,1)
        text(reg_day_cents(i,1),reg_day_cents(i,2),num2str(i),'HorizontalAlignment','center');
        end
    end 
    title('Registering day')
    
    subplot(2,2,2)
    plot(bsxfun(@plus,1:size(regcells_temp_day.dFF,2),regcells_temp_day.dFF))
    
    subplot(2,2,4)
    plot(bsxfun(@plus,1:size(regcells_reg_day.dFF,2),regcells_reg_day.dFF))
end
%% file_1_reg

file_1_reg = insertAfter(file_1,['F'],[days_involved '_registered_all_days']);
save([path_1 file_1_reg],'-struct','regcells_reg_day');


%% file_2_reg

file_2_reg = insertAfter(file_2,['F'],[days_involved '_registered_all_days']);
save([path_2 file_2_reg],'-struct','regcells_temp_day');

