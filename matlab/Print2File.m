function Print2File( U, seq, fileName )

fileID = fopen(fileName, 'w');

% N = size(U, 1);
M = size(U, 2);

prtStr = '%d';
for d = 1:M
    prtStr = strcat(prtStr, ' %f');
end

fprintf(fileID, '#%d \n', M);

for i = 1:length(seq)
    Ui = U(i, :)';
    
    % add id
    Ui = cat(1, seq(i), Ui);
    
    fprintf(fileID, prtStr, Ui);
    
    if(i ~= length(seq))
        fprintf(fileID, '\n');
    end
end

fclose(fileID);

end

 