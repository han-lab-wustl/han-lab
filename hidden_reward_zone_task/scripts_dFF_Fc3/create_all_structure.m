% calculate dff and Fc3
function all = create_all_structure(F,Fneu,spks,Fs,cells)
% clear the environment and load your neural and behavioral data aligned
    
    % outputs
        F = F(cells,:);
        Fneu = Fneu(cells,:);
        spks = spks(cells,:);
        if numel(Fs)~=1
            Fs = [];
            Fs = 32; %Hz
        end
        [dFF,Fc3] = Calc_Fc3_Reverse_Subtraction(F,Fneu,Fs);

        dFF = dFF';
        Fc3 = Fc3';

    
    % save neuronal data for the whole session
    all.dff = dFF;
    all.Fc3 = Fc3;
    all.Spks = spks;
  
end
    