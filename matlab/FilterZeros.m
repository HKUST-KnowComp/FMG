function [ Spa, rMap, cMap ] = FilterZeros( Spa )

nnzRow = full(sum(Spa, 2));
nnzCol = full(sum(Spa, 1));

rMap = find(nnzRow ~= 0);
cMap = find(nnzCol ~= 0)';

Spa = Spa(nnzRow ~= 0, :);
Spa = Spa(:, nnzCol ~= 0);

end

