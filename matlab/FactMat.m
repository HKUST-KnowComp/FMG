clear;
clc;

dirName = 'yelp-mat';
outDirt = 'yelp-dat';

fileList = dir(dirName);

% lambda_0 = 0.25e-4; % amazon
lambda_0 = 1e-4; % yelp

para.tol = 1e-6;
para.maxR = 50;
para.maxIter = 200;

for i = 1:length(fileList)
    
    fileName = fileList(i).name;

    if(strfind(fileName, '.mat') >= 1)
        data = load(strcat(dirName, '/', fileName));
        data = data.data;

        row = data(:, 1);
        col = data(:, 2);
        val = data(:, 3);

        val = val - mean(val);

        lambda_i = lambda_0*length(val);
        
        Spa = sparse(row, col, val);
        [ Spa, rMap, cMap ] = FilterZeros( Spa );

        [U, S, V, output ] = AISImpute( Spa, lambda_i, para );

        S = diag(S);
        S = sqrt(S);

        U = U*diag(S);
        V = V*diag(S);
        
        figure;
        semilogy(output.Time, output.obj);
        title(num2str(i))
        xlabel(strcat(  num2str(length(S)) ));

        
        % print to file
        newName_user = strrep(fileName, '.mat', '_user.dat');
        Print2File( U, rMap, strcat(outDirt, '/', newName_user) );
        
        newName_item = strrep(fileName, '.mat', '_item.dat');
        Print2File( V, cMap, strcat(outDirt, '/', newName_item) );
    end

end



