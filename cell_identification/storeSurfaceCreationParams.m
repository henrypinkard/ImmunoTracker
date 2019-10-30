%run this script to store the values of all the surface creation params for
%a particular cell population setting
cellPopName = 'gfp';

saveCellTypeParams(cellPopName, 'channelIndex', 2);
saveCellTypeParams(cellPopName, 'smoothFilter', 1);
saveCellTypeParams(cellPopName, 'localContrast', 8);
saveCellTypeParams(cellPopName, 'backgroundSub', 10);
saveCellTypeParams(cellPopName, 'seedDiam', 12); 
saveCellTypeParams(cellPopName, 'quality', 2);
saveCellTypeParams(cellPopName, 'minNumVoxels', 70);

cellPopName = 'rfp';

saveCellTypeParams(cellPopName, 'channelIndex', 4);
saveCellTypeParams(cellPopName, 'smoothFilter', 1);
saveCellTypeParams(cellPopName, 'localContrast', 8);
saveCellTypeParams(cellPopName, 'backgroundSub', 10);
saveCellTypeParams(cellPopName, 'seedDiam', 12); 
saveCellTypeParams(cellPopName, 'quality', 2);
saveCellTypeParams(cellPopName, 'minNumVoxels', 70);

cellPopName = 'e670';

saveCellTypeParams(cellPopName, 'channelIndex', 5);
saveCellTypeParams(cellPopName, 'smoothFilter', 1);
saveCellTypeParams(cellPopName, 'localContrast', 8);
saveCellTypeParams(cellPopName, 'backgroundSub', 10);
saveCellTypeParams(cellPopName, 'seedDiam', 12); 
saveCellTypeParams(cellPopName, 'quality', 2);
saveCellTypeParams(cellPopName, 'minNumVoxels', 20);

cellPopName = 'vpd';

saveCellTypeParams(cellPopName, 'channelIndex', 1);
saveCellTypeParams(cellPopName, 'smoothFilter', 1);
saveCellTypeParams(cellPopName, 'localContrast', 8);
saveCellTypeParams(cellPopName, 'backgroundSub', 4);
saveCellTypeParams(cellPopName, 'seedDiam', 12); 
saveCellTypeParams(cellPopName, 'quality', 1);
saveCellTypeParams(cellPopName, 'minNumVoxels', 40);

cellPopName = 'xcr1';

saveCellTypeParams(cellPopName, 'channelIndex', 3);
saveCellTypeParams(cellPopName, 'smoothFilter', 1);
saveCellTypeParams(cellPopName, 'localContrast', 14);
saveCellTypeParams(cellPopName, 'backgroundSub', 4);
saveCellTypeParams(cellPopName, 'seedDiam', 14); 
saveCellTypeParams(cellPopName, 'quality', 1);
saveCellTypeParams(cellPopName, 'minNumVoxels', 50);

% else if strcmp(cellTypeName, 'e670') %TODO improve this
%     channelIndex = 5;
%     smoothFilter = 1.5;
%     localContrast = 10;
%     backgroundSub = 10;
%     seedDiam = 12;
%     quality = 1;
%     minNumVoxels = 150;  

