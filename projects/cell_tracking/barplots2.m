%% Mouse e201 Totals week to day
figure;
one = 39;
two = 5;
three = 11;
four = 0;

y = [one,two,three,four]
b = bar(y)
xticklabels({'Category 1: cell does not fire','Category 2: Arifact','Category 3: No cell tracking match','Category 4: no idea where it is'})
ylabel('Number of Cells');
title('Cell Tracking for Week12 Missing in Days 55-59 for Mouse e201');
%annotation('textbox',[0.75,0.1,0.1,0.1],'String',"Category 1: cell does not fire, " + ...
 %   "Category 2: Arifact, Category 3: No cell tracking match, Category 4: no idea where it is");
%% Mouse 1 totals, e186
figure;
one = 144;
two = 0;
three = 26;
four = 6;
;
y = [one,two,three,four]
b = bar(y)
xticklabels({'Category 1: cell does not fire','Category 2: Arifact','Category 3: No cell tracking match','Category 4: no idea where it is'})
ylabel('Number of Cells');
title('Cell Tracking for Week1 Missing in Days 1-4 for Mouse e186');

%% Mouse 1 and 2 combined for week to day totals
figure;
one = 183;
two = 5;
three = 37;
four = 6;
;
y = [one,two,three,four]
b = bar(y)
xticklabels({'Category 1: cell does not fire','Category 2: Arifact','Category 3: No cell tracking match','Category 4: no idea where it is'})
ylabel('Number of Cells');
title('Combined Cell Tracking for Mouse e201 and e186 for Cells Present in Weekly but Missing in Daily');
%% B
