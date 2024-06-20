function com = calc_COM_EH(spatial_act,bin_width) 
%spatial_act is #cells x #bins
%spatial_act=tuning curve
% bin_width in cm

%added omitNaN to sum functions

bin = zeros(size(spatial_act,1),1);%1st bin above mid pt
frac = zeros(size(spatial_act,1),1);%fraction for interpolated COM
com = zeros(size(spatial_act,1),1);%interpolated COM in cm
sum_spatial_act=sum(spatial_act,2);%get total fl. from tuning curve
mid_sum=sum_spatial_act/2;%mid point of total fl.
spatial_act_cum_sum=cumsum(spatial_act,2);%cumulative sum of fl in tuning curve
idx_above_mid=spatial_act_cum_sum>=mid_sum;%logical of indexes above mid fl
for i = 1:size(spatial_act,1)
    if ~isnan(sum_spatial_act(i))
    
%     com(i,1) = find(idx_above_mid(i,:),1,'first')*bin_width;%find index of first bin above mid fl. 
    bin(i,1) = find(idx_above_mid(i,:),1,'first');%find index of first bin above mid fl. 
    %linear interp
    if bin(i,1) ==1 %if mid point is in 1st bin
        frac(i,1)=(spatial_act_cum_sum(i,bin(i,1))-mid_sum(i,1))/(spatial_act_cum_sum(i,bin(i,1)));%don't need spatial_act_cum_sum(i,(bin(i,1)-1)
        com(i,1)= (frac(i,1))*bin_width;%only need fraction       
    else %don't think i need specific case for last bin
        frac(i,1)=(spatial_act_cum_sum(i,bin(i,1))-mid_sum(i,1))/(spatial_act_cum_sum(i,bin(i,1))-(spatial_act_cum_sum(i,(bin(i,1)-1))));%frac of mid to fl in bin
        com(i,1)= ((bin(i,1)-1)+frac(i,1))*bin_width;%add the fraction to last bin before going over * bin_width = interpolated com
    end
    else
        
        com(i,1) = NaN;
    end
    
end


