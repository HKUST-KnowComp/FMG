clear; clc;

dirName = 'yelp-txt';

fileList = dir(dirName);

for i = 1:length(fileList)
    fileName = fileList(i).name;
    
    if(strfind(fileName, '.res') >= 1)
        
        data = load( strcat(dirName, '/', fileName) );
        
        save(strrep(fileName, '.res', '.mat'), 'data');
    end
end