% % sumEpochMaxes.m
% % -------------------------------------------------------------------------
% % Create toy data: zeros between epochs, epochs labelled 1,2,3…; then
% % detect each epoch, take its max (the trial number), and sum those.
% % -------------------------------------------------------------------------
% 
% clear; clc;
% 
% %% PARAMETERS
% N          = 4000;   % total length of the vector
% numTrials  = 22;     % how many epochs/trials to simulate
% epochLen   = 15;     % samples per epoch
% zeroLen    = 5;      % zeros between epochs
% 
% %% 1) BUILD TOY VECTOR
% data = zeros(1, N);
% pos  = 1;
% for t = 1:numTrials
%     % advance past zeros
%     pos = pos + zeroLen;
%     if pos + epochLen - 1 > N, break; end
% 
%     % fill one epoch with the trial number t
%     data(pos : pos + epochLen - 1) = t;
%     pos = pos + epochLen;
% end

data = trialnumALL;
%% 2) FIND EACH NONZERO EPOCH
mask = data ~= 0;
d    = diff([0, mask, 0]);          % pad to catch first/last run
starts = find(d == +1);             % epoch starts
ends   = find(d == -1) - 1;         % epoch ends
nEpochs = numel(starts);

%% 3) EXTRACT MAX OF EACH EPOCH & SUM
epochMax = arrayfun(@(i) max(data(starts(i):ends(i))), 1:nEpochs);
totalSum = sum(epochMax);

%% 4) DISPLAY
fprintf('Detected %d epochs.\n', nEpochs);
fprintf('Max per epoch: [%s]\n', num2str(epochMax));
fprintf('Sum of all epoch-maxima = %d\n', totalSum);

%% 5) OPTIONAL VISUALIZATION
figure; 
plot(data, '.', 'MarkerSize', 8);
ylim([-1, numTrials+2]);
xlabel('Sample Index');
ylabel('Value');
title('Toy Data: zeros + ascending-trial-number epochs');
grid on;
