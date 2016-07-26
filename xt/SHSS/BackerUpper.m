
selectionName = 'DCs';
%get file
[filename, pathname] = uigetfile('*.mat');
if (filename == 0)
    return;
end

surfFile = matfile(strcat(pathname,filename));
eval(sprintf('%s = surfFile.%s',selectionName, selectionName))
save(strcat('D:\Data\Henry\SHSS selection backups\',selectionName,'.mat'),selectionName )
