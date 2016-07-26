clear
load('rawData.mat')
ii = statistics(1).Ids(TCellIndices + 1);

%get file
filename =  '/Users/henrypinkard/Desktop/LNData/CMTMRCandidates.mat';
[xImarisApp xPreviewSurface] = xtSetupSurfaceTransfer();
surfFile = matfile(filename,'Writable',false);

% func_addsurfacestosurpass(xImarisApp,surfFile,1,xPreviewSurface,ii);

%delete incorrect ones by hand