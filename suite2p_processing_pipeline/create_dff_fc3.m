function [fallpth, dFF, Fc3] = create_dff_fc3(fallpth, Fs)
% zahra's function for fc3 and dff
% includes no filters, including no iscell or skewness
load(fallpth, 'F', 'Fneu')
[dFF,Fc3] = Calc_Fc3_Reverse_Subtraction(F,Fneu,Fs);
save(fallpth, 'dFF', 'Fc3', '-append')
end