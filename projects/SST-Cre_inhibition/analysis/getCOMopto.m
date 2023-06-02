function [com] = getCOMopto(comfl, condind)

% get COM (of successful trials only)
% across multiple days
load(comfl)
% remove blank days
COM = COM(~cellfun('isempty',COM));
% init empty array for conditions that dont count
init = cell(1,length(COM));
init(1:length(COM))={NaN(1,5)};
% second row (day 2 of condition), 3rnd column (epoch #1)
if condind==1
    COMep2=cellfun(@(x) x{1}, cellfun(@(x) x(2,1), COM, 'UniformOutput', false), 'UniformOutput', false);
    % first 5 trials of ep 2 ('optoed')
    com{1} = cellfun(@(x) x(1:5), COMep2, 'UniformOutput', false);
    com{2} = cellfun(@(x) x(6:end), COMep2, 'UniformOutput', false);
    com{5} = init;
    com{6} = init;
    COMep3=cellfun(@(x) x{1}, cellfun(@(x) x(3,1), COM, 'UniformOutput', false), 'UniformOutput', false);
    try
        com{7} = cellfun(@(x) x(1:5), COMep3, 'UniformOutput', false);
        com{8} = cellfun(@(x) x(6:end), COMep3, 'UniformOutput', false);
    catch %  assume it failed bc not enough trials for now (eg trials < 5)
        com{7} = cellfun(@(x) x(1:end), COMep3, 'UniformOutput', false);
        com{8} = init;
    end
elseif condind==2
    COMep3=cellfun(@(x) x{1}, cellfun(@(x) x(3,1), COM, 'UniformOutput', false), 'UniformOutput', false);
    % first 5 trials of ep 3 ('optoed')
    try
        com{1} = cellfun(@(x) x(1:5), COMep3, 'UniformOutput', false);
        com{2} = cellfun(@(x) x(6:end), COMep3, 'UniformOutput', false);
    catch
        com{1} = cellfun(@(x) x(1:end), COMep3, 'UniformOutput', false);
        com{2} = init;
    end
    COMep2=cellfun(@(x) x{1}, cellfun(@(x) x(2,1), COM, 'UniformOutput', false), 'UniformOutput', false);
    com{5} = cellfun(@(x) x(1:5), COMep2, 'UniformOutput', false);
    com{6} = cellfun(@(x) x(6:end), COMep2, 'UniformOutput', false);
    com{7} = init;
    com{8} = init;
else % if control days
    com{1} = init;
    com{2} = init;
    COMep2=cellfun(@(x) x{1}, cellfun(@(x) x(2,1), COM, 'UniformOutput', false), 'UniformOutput', false);
    com{5} = cellfun(@(x) x(1:5), COMep2, 'UniformOutput', false);
    com{6} = cellfun(@(x) x(6:end), COMep2, 'UniformOutput', false);
    COMep3=cellfun(@(x) x{1}, cellfun(@(x) x(3,1), COM, 'UniformOutput', false), 'UniformOutput', false);
    com{7} = cellfun(@(x) x(1:5), COMep3, 'UniformOutput', false);
    com{8} = cellfun(@(x) x(6:end), COMep3, 'UniformOutput', false);
end
% ep 1 is the same throughout
COMep1=cellfun(@(x) x{1}, cellfun(@(x) x(1,1), COM, 'UniformOutput', false), 'UniformOutput', false);
com{3} = cellfun(@(x) x(1:5), COMep1, 'UniformOutput', false);
com{4} = cellfun(@(x) x(6:end), COMep1, 'UniformOutput', false);

end