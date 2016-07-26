%%export selected indices + statistics for further analysis
clear
selectionName = 'DCs';
%get file
[filename, pathname] = uigetfile('*.mat');
if (filename == 0)
    return;
end

surfFile = matfile(strcat(pathname,filename));
eval(sprintf('%s = surfFile.%s',selectionName, selectionName))
stats = surfFile.stats;
save(strcat('C:\Users\hpinkard\Dropbox\Henry\Papers\LN imaging paper\surface detector\',selectionName,'.mat') )