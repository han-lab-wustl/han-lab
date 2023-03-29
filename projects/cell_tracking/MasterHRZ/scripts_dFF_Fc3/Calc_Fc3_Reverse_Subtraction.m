
function [dFF,Fc3] = Calc_Fc3_Reverse_Subtraction(F,Fneu,Fs,varargin)
% Returns Fc3 and dFF for all the F traces given in imput

% IMPUTs
% F1 is the raw fluorescence signal, each row is one cell
% Fneurol is the extracted neuropil, each row is one cell
% Fs is the frame rate of your recording (Note: 32/n of frames)
% 4th imput can be the 'iscell' variable
% 5th imput can be the time window to calculate dFF, if not specified = 25s


% OUTPUTs
% dFF, percentange change from baseline
% Fc3, identified transients based on the ration between negative and 
% positive transients of each individual cell.

% REQUIRED FUNCTIONS 
% Paths:
% C:\Users\Han lab\Documents\MATLAB\Matlab_scripts\place_cell_scripts\image_analysis
% C:\Users\Han lab\Documents\MATLAB\HANLAB_SCRIPTS-master\Matlab_scripts\place_cell_scripts
% C:\Users\Han lab\Documents\MATLAB\EB scripts\HRZ basic functions
% Functions:
% redo_dFF
% find_min_signif_dur_Fpolyv3 ___Note: 'v3' is the latest version
% calc_Fc3v3

% GM EB 4/20/21 
% gmolina@wustl.edu
% eleonora.bano@wustl.edu
% *************************************************************************

% Rename imput variables
F1 = F;
Fneuro1 = Fneu;
dff_window = 25;

if nargin > 3 
    if ~isempty(varargin{1})
    % variable defining the selected cells
    iscell = varargin{1};
    F1 = F(iscell(:,1)==1,:);
    Fneuro1 = Fneu(iscell(:,1)==1,:);
    end
end

if nargin > 4 
    % predefine time window to calculate dFF
    dff_window = varargin{2};
end

% calc dFF without baseline subtraction
dFFnc = redo_dFF(F1',Fs,dff_window);

% find the ratio of positive going transients and negative going transients
% for each cell
std_min_sig_dur = find_min_signif_dur_Fpolyv3(dFFnc);

% calc dFF with baseline subtraction
dFF = redo_dFF(F1',Fs,dff_window,Fneuro1');

%use the ratio calculated using dFFnc to identify transients in dFF
Fc3 = calc_Fc3v3(dFF,std_min_sig_dur);

%ViewResults

% % Plot in a slideshow like format the identified traces
% figure;
% % plot dFF traces and Fc3
% for i = 1:size(Fc3,2)
%     plot(dFFc(15000:20000,i))
%     pause(0.5)
%     hold on
%     plot(Fc3(15000:20000,i))
%     pause(0.8)
%     hold off
% end
end
