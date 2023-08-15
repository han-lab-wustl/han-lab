function [dff, fc3] = skewness_filter(dff, fc3)
% zd added to optionally removed skewed cells
skews=[];
skews = skewness(dff,1,2);
dff(skews<2,:) = [];
fc3(skews<2,:) = [];
end