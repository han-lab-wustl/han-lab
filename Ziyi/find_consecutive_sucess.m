epoch_start = find_first_consecutive_numbers(find(trialnum == 0));
epoch_start_frame = [1 epoch_start length(trialnum)];
epoch_period = cell(1, length(epoch_start_frame) - 1);

for i = 1:length(epoch_period)
    epoch_period{i} = [epoch_start_frame(i) epoch_start_frame(i+1) - 1];
end

consecutive_success_csIndx = [];

for i = 1:length(epoch_period)
    single_epoch_start = epoch_period{i}(1);
    single_epoch_end = epoch_period{i}(2);
    single_epoch_trialnum = trialnum(single_epoch_start:single_epoch_end);
    single_epoch_cs = solenoid2(single_epoch_start:single_epoch_end);
    csNum = find(single_epoch_cs);
    consecutive_success_indx = find_indices_with_consecutive_before(single_epoch_trialnum(csNum));
    single_epoch_consecutive_success_csIndx = csNum(consecutive_success_indx) + single_epoch_start - 1;
    consecutive_success_csIndx = [consecutive_success_csIndx single_epoch_consecutive_success_csIndx];
end
