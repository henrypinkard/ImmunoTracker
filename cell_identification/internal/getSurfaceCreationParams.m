function [channelIndex, smoothFilter, localContrast, backgroundSub, seedDiam, quality,...
    minNumVoxels] = getSurfaceCreationParams(cellTypeName)
%GETCELLTYPEPARAMS Returns needed for surface creation.
%StoreSurfaceCreationParams.m needs to have been run before calling this
saveFile = matfile(sprintf('CellTypeParams_%s',cellTypeName),'Writable',false);

channelIndex = saveFile.channelIndex;
smoothFilter = saveFile.smoothFilter;
localContrast = saveFile.localContrast;
backgroundSub = saveFile.backgroundSub;
seedDiam = saveFile.seedDiam;
quality = saveFile.quality;
minNumVoxels = saveFile.minNumVoxels;
    
end

