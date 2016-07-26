function [] = adddistancetosurfacestat(surfFile, transformFile)

%read info from file
numTimePoints = size(transformFile,'timeIndices',1);
refSurfaceName = transformFile.surfaceName;
pixelSizeXY = transformFile.pixelSizeXY;
pixelSizeZ = transformFile.pixelSizeZ;

%preallocate array for speed
numSurfaces = size(surfFile,'timeIndex',1);
distances = zeros(numSurfaces,1);
numVertices = surfFile.numVertices;
maskTimeIndices = transformFile.timeIndices;
currentTransform = -1;
tic
for timeIndex = 0:numTimePoints-1;
    toc
    fprintf('Time point %i of %i\n',timeIndex+1,numTimePoints);
    %get transform closest to time index
    indicesFromTransform = abs(find(maskTimeIndices) - timeIndex - 1);
    transformToUseIndex = find(min(indicesFromTransform) == indicesFromTransform,1);
    if (currentTransform ~= transformToUseIndex)
        currentTransform = transformToUseIndex;
        fprintf('loading distance transform\n');
        distanceTransform = transformFile.distanceTransform(:,:,:,transformToUseIndex);
        toc
    end
    %     go through all surfaces, find centroid, convert to pixels, get value
    %     in distance transform, store
    currentTPIndices = find(surfFile.timeIndex == timeIndex);
    for index = currentTPIndices';
        verticesOffset = sum(numVertices(1:index - 1)) + 1;
        surfCenter = mean(surfFile.vertices(verticesOffset:verticesOffset+numVertices(index) - 1,:));
        pixelIndices = floor(surfCenter ./ [pixelSizeXY pixelSizeXY pixelSizeZ]) + [1 1 1];
        distances(index) = distanceTransform(pixelIndices(1), pixelIndices(2), pixelIndices(3));
    end
end
%get stat struct
stats = surfFile.stats;
%prepare in statStruct format
distStat = struct('Ids',stats(1).Ids,'Name',['Distance to ' char(refSurfaceName)]...
    ,'Values',single(distances),'Units','um');
%add to stats
stats(length(stats) + 1) = distStat;
%sort into alphabetical order
[~, statOrder] = sort({stats.Name});
stats = stats(statOrder);
%resave
surfFile.stats = stats;
end