<<<<<<< HEAD

clear all 
days = input('Total number of days to register (including template day)');
plane = input('Plane that you are registering');
days_involved = input('Enter days involved - ex : _D9_D10_D11_D12_D13 \n');



disp('For all days...')
for j = 1:days-1
    ndays{j} = input('Which day are you about to load (column number in the registered file that has all days,\n i.e. if the file is D3_D4_D5 D3 = 1.   :');
    [file_1{j},path_1{j}] = uigetfile('*F.mat',['Select the registering day number ',num2str(ndays{j}),'(IN THE REG FILE) -F file for plane ',num2str(plane)]);
    filename_1{j} = [path_1{j} file_1{j}];
    

    
end

[file_2,path_2] = uigetfile('*F.mat',['Select the template days file F for plane ',num2str(plane)]');
filename_2 = [path_2 file_2];

    [file_3,path_3] = uigetfile('*all_days.mat',['Select the registered file that has all days for plane ',num2str(plane)]');
    filename_3 = [path_3 file_3];
% planes = input('Total number of planes');
for i = 1
    for j = 1:days-1
        nday = ndays{j};
        
%     fprintf('Plane number is %i\n', plane)
%     fprintf('Registering Day number is %i\n', nday)
        
        get_registered_cells_Suite2p_multiple_days(plane,nday,days_involved,filename_1{j},filename_2,filename_3,file_1{j},path_1{j},file_2,path_2);
    end 
=======

clear all 
days = input('Total number of days to register (including template day)');
plane = input('Plane that you are registering');
days_involved = input('Enter days involved - ex : _D9_D10_D11_D12_D13 \n');



disp('For all days...')
for j = 1:days-1
    ndays{j} = input('Which day are you about to load (column number in the registered file that has all days,\n i.e. if the file is D3_D4_D5 D3 = 1.   :');
    [file_1{j},path_1{j}] = uigetfile('*F.mat',['Select the registering day number ',num2str(ndays{j}),'(IN THE REG FILE) -F file for plane ',num2str(plane)]);
    filename_1{j} = [path_1{j} file_1{j}];
    

    
end

[file_2,path_2] = uigetfile('*F.mat',['Select the template days file F for plane ',num2str(plane)]');
filename_2 = [path_2 file_2];

    [file_3,path_3] = uigetfile('*all_days.mat',['Select the registered file that has all days for plane ',num2str(plane)]');
    filename_3 = [path_3 file_3];
% planes = input('Total number of planes');
for i = 1
    for j = 1:days-1
        nday = ndays{j};
        
%     fprintf('Plane number is %i\n', plane)
%     fprintf('Registering Day number is %i\n', nday)
        
        get_registered_cells_Suite2p_multiple_days(plane,nday,days_involved,filename_1{j},filename_2,filename_3,file_1{j},path_1{j},file_2,path_2);
    end 
>>>>>>> 754f532e47d152334ffae033cf3e5763ab9bf2c0
end 