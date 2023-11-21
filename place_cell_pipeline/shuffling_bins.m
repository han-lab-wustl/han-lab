
function big_shuffbin = shuffling_bins(X)

timepts_nonzero = find(X~=0);
timepts_zero = find(X == 0);

if isempty(find(X == 0))
    timepts_zero = 1:length(X);
    clear timepts_nonzero
    timepts_nonzero = [];
end

transient_locs = consecutive_stretch(timepts_nonzero);
zerobin_size = 5;
n_full_zerobins = floor(length(timepts_zero)/zerobin_size);
for i = 1:floor(length(timepts_zero)/zerobin_size)
    shuff_bin{i} = timepts_zero(zerobin_size*(i-1)+1 : zerobin_size*i);
end

if n_full_zerobins > 0
    shuff_bin{i+1} = timepts_zero(n_full_zerobins*zerobin_size+1 : length(timepts_zero));
else
    shuff_bin{1} = timepts_zero(n_full_zerobins*zerobin_size+1 : length(timepts_zero));
end

big_shuffbin = [shuff_bin,transient_locs];
for i = 1:length(big_shuffbin)
    l(i) = length(big_shuffbin{i});
end

delete_idx = find(l == 0);
big_shuffbin(delete_idx) = [];


% clearvars -except mouse_data
end

% count = 0;
% for i = 1:length(big_shuffbin)
%     count = count + length(big_shuffbin{i});
% end
