function newCell = rightAlignCell(C)
%function made to take a 2d-cell with several entries (some empty) and right align instead
%of left align eg:
%orginal Cell Matrix: 7x8 cell
%{79x48 double,79x48 double,79x48 double,79x48 double,[],[],[],[];
%79x18 double,79x18 double,79x18 double,79x18 double,[],[],[],[];
%79x50 double,79x50 double,79x50 double,79x50 double,79x50 double,[],[],[];
%79x51 double,79x51 double,79x51 double,79x51 double,79x51 double,79x51 double,[],[];
%79x50 double,79x50 double,79x50 double,79x50 double,[],[],[],[];
%79x49 double,79x49 double,79x49 double,79x49 double,[],[],[],[];
%79x39 double,79x39 double,79x39 double,79x39 double,79x39 double,79x39 double,79x39 double,79x39 double}

%output Cell Matrix: 7x8 cell
%{[],[],[],[],79x48 double,79x48 double,79x48 double,79x48 double;
%[],[],[],[],79x18 double,79x18 double,79x18 double,79x18 double;
%[],[],[],79x50 double,79x50 double,79x50 double,79x50 double,79x50 double;
%[],[],79x51 double,79x51 double,79x51 double,79x51 double,79x51 double,79x51 double;
%[],[],[],[],79x50 double,79x50 double,79x50 double,79x50 double;
%[],[],[],[],79x49 double,79x49 double,79x49 double,79x49 double;
%79x39 double,79x39 double,79x39 double,79x39 double,79x39 double,79x39 double,79x39 double,79x39 double}

if ~iscell(C)
    error('Input Must be in the form of a cell')
end

numi = size(C,1);
numj = size(C,2);
newCell = cell(size(C));


for i = 1:numi
    emptyrows = cellfun(@isempty,C(i,:));
    for j = 1:numj-sum(emptyrows)
        newCell(i,j+sum(emptyrows)) = C(i,j);
    end
end