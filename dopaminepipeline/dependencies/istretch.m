function [C]= istretch(a,tmfr)
clear C
i1 = 1;

C{i1}=a(i1);
for j1 = 2:numel(a)
    t = a(j1)-a(j1-1);
    if t == 1
        C{i1} = [C{i1} a(j1)];
    else
        i1  = i1 + 1;
        C{i1} = a(j1);
    end
end

end