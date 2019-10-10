function [] = createSurfaces(paramsKey)
%CREATESURFACES
% Run with an open dataset in Imaris 7.6.5. paramsKey corresponds to what
% was used for storeSurfaceCreationParams.m
%creates and saves surfaces and statistics, reads image data and masks

%change to folder above
cd(fileparts(mfilename('fullpath')))

%Parameters that can be tuned to optimize performance
batchSize = 100;
framesPerLoop = 1; %number of frames for which surfaces are created with each loop

%add java libraries to dynamic path
javaaddpath('./ImarisLib.jar');

getSurfaceCreationParams(paramsKey);
[channelIndex, smoothFilter, localContrast, backgroundSub, seedDiam, quality,...
    minNumVoxels] = getSurfaceCreationParams(paramsKey);


%connect to imaris
imarisIndex = 0;
vImarisLib = ImarisLib;
imaris = vImarisLib.GetApplication(imarisIndex);
if (isempty(imaris))
    msgbox('Wrong imaris index');
    return;
end

% Create matlab file to save surface data in
filename = imaris.GetCurrentFileName;
imageProcessing = imaris.GetImageProcessing;
ds = imaris.GetDataSet;
fovSize = [ds.GetExtendMaxX ds.GetExtendMaxY];

numTimePoints = ds.GetSizeT;
imageWidth = ds.GetSizeX;
imageHeight = ds.GetSizeY;
pixelSizeXY = ds.GetExtendMaxX / ds.GetSizeX;
pixelSizeZ = ds.GetExtendMaxZ / ds.GetSizeZ;

%generate saving name
imsFileFullPath = strsplit(char(filename),'.');
imsFilePathAndName = imsFileFullPath{1};
%save in same directory as data
saveName = strcat(imsFilePathAndName,'_',paramsKey,'_candidates','.mat');
%delete exisiting file with same name
if (exist(saveName,'file') == 2)
    if (strcmp(questdlg('Delete exisiting file'),'Yes'))
        delete(saveName);
    end
end
saveFile = matfile(saveName,'Writable',true);
if ~any(strcmp('vertices',who(saveFile)))
    %initialize fields
    saveFile.stitchedXYZPositions = single([]);
    saveFile.excitations = single([]);
    saveFile.masks = {};
    saveFile.imageData = {};
    saveFile.imarisIndices = {};
    startIndex = 0;
    saveFile.vertices = single([]);
    saveFile.triangles = int32([]);
    saveFile.normals = single([]);
    saveFile.timeIndex = int32([]);
    saveFile.numTriangles = int32([]);
    saveFile.numVertices = int32([]);
    saveFile.name = paramsKey;
    saveFile.surfInterpPoints = {};
    saveFile.pixelSizeXY = pixelSizeXY;
    saveFile.pixelSizeZ = pixelSizeZ;

    offsets = zeros(6, 1);
    for c  = 0:5
         pixels = typecast(ds.GetDataSubVolumeAs1DArrayBytes(0, 0, 0, c, 0,...
             min(1200, ds.GetSizeX), min(1200, ds.GetSizeY),...
             min(100, ds.GetSizeZ)), 'uint8');
        nonzeropix = pixels(pixels ~= 0);
        offsets(c + 1) = prctile(nonzeropix, 25);
    end
    saveFile.channelOffsets = offsets;
else
    startIndex = saveFile.lastWrittenFrameIndex + 1;
end


%Incrementally segment surfaces, get and save their info, and delete
maxFrameIndex = numTimePoints-1;
for startFrame = startIndex:framesPerLoop:maxFrameIndex
    tic
    fprintf('Calculating surfaces on frame %i-%i of %i\n',startFrame, min(startFrame + framesPerLoop-1,maxFrameIndex),maxFrameIndex);
    % time index is 0 indexed and inclusive, but shows up in imaris as 1
    % indexed
    roi = [0,0,0,startFrame,ds.GetSizeX,ds.GetSizeY,ds.GetSizeZ,min(startFrame + framesPerLoop-1,maxFrameIndex)];
    
    surface = imageProcessing.DetectSurfacesRegionGrowing(ds, roi,channelIndex,smoothFilter,localContrast,false,backgroundSub,...
        seedDiam,true, sprintf('"Quality" above %1.3f',quality),sprintf('"Number of Voxels" above %i',minNumVoxels));
    %only process if there are actually surfaces
    if surface.GetNumberOfSurfaces <= 0
        surface.RemoveAllSurfaces;
        continue;
    end
    
    %get stats
    stats = xtgetstats(imaris, surface, 'ID', 'ReturnUnits', 1);
    %modify stats to reflect stitching
%     stats = modify_stats_for_stitched_view(stats);
    
    % iterate through batches of segmeneted surfaces (so as to not overflow
    % memory),
    for i = 0:ceil((surface.GetNumberOfSurfaces)/batchSize ) - 1
        startIndex = batchSize*i;
        endIndex = min(surface.GetNumberOfSurfaces-1, batchSize*(i+1) - 1);
        fprintf('%d to %d of %d\n',startIndex+1, endIndex +1, surface.GetNumberOfSurfaces)
        surfList = surface.GetSurfacesList(startIndex:endIndex);
        %stitch surfaces
%         [vertices, tIndices] = modify_surfaces_for_stitched_view(surfList.mVertices, surfList.mTimeIndexPerSurface, surfList.mNumberOfVerticesPerSurface,...
%             posList, imageWidth, imageHeight, xPixelOverlap, yPixelOverlap, pixelSizeXY, numTimePoints);
        
        maskTemp = cell(endIndex-startIndex+1,1);
        imgTemp = cell(endIndex-startIndex+1,1);
        %iterate through each surface to get its mask and image data
        for index = startIndex:endIndex
            %get axis aligned bounding box
            boundingBoxSize = [stats(find(strcmp({stats.Name},'BoundingBoxAA Length X'))).Values(index + 1),...
                stats(find(strcmp({stats.Name},'BoundingBoxAA Length Y'))).Values(index + 1),...
                stats(find(strcmp({stats.Name},'BoundingBoxAA Length Z'))).Values(index + 1)];
            positionUnstitched = [stats(find(strcmp({stats.Name},'Position X'))).Values(index + 1),...
                stats(find(strcmp({stats.Name},'Position Y'))).Values(index + 1),...
                stats(find(strcmp({stats.Name},'Position Z'))).Values(index + 1)];           
            
            %Mask is in unstitched coordinates, image data is in stitched coordinates
            topLeft = positionUnstitched -  boundingBoxSize / 2;
            bottomRight = positionUnstitched + boundingBoxSize / 2;
            
            topLeftPix = int32(floor([topLeft(1:2) ./ pixelSizeXY topLeft(3) ./ pixelSizeZ]));
            bottomRightPix = int32(ceil(bottomRight ./ [pixelSizeXY pixelSizeXY pixelSizeZ]));
            maxExtent = [ds.GetSizeX, ds.GetSizeY, ds.GetSizeZ];
            if any(bottomRightPix > maxExtent)
               bottomRightPix = min(bottomRightPix, int32(maxExtent));
            end
            if any(topLeftPix > 0)
               topLeftPix = max(topLeftPix, int32([0 0 0]));
            end
            pixelResolution = bottomRightPix - topLeftPix;
            
            
            mask = surface.GetSingleMask(index, topLeft(1), topLeft(2), topLeft(3), bottomRight(1), bottomRight(2),...
                bottomRight(3), pixelResolution(1), pixelResolution(2), pixelResolution(3));
            byteMask = uint8(squeeze(mask.GetDataBytes));
            timeIndex = surface.GetTimeIndex(index);
            imgData = uint8(zeros(pixelResolution(1),  pixelResolution(2), pixelResolution(3), 6));
            for c = 0:5
                signed = ds.GetDataSubVolumeBytes(topLeftPix(1), topLeftPix(2), topLeftPix(3), c, timeIndex,...
                pixelResolution(1), pixelResolution(2), pixelResolution(3));
                imgData(:,:,:,c+1) = reshape(typecast(signed(:),'uint8'), size(signed));
            end

            
            %trim to border of mask
            maskBorder = false(size(byteMask));
            zMin = find(squeeze(sum(sum(byteMask,1),2)),1);
            zMax = find(squeeze(sum(sum(byteMask,1),2)),1,'last');
            xMin = find(squeeze(sum(sum(byteMask,2),3)),1) ;
            xMax = find(squeeze(sum(sum(byteMask,2),3)),1,'last') ;
            yMin = find(squeeze(sum(sum(byteMask,1),3)),1) ;
            yMax = find(squeeze(sum(sum(byteMask,1),3)),1,'last');
            maskBorder(xMin:xMax,yMin:yMax,zMin:zMax) = 1;
            byteMask = reshape(byteMask(maskBorder), xMax-xMin+1, yMax-yMin+1, zMax-zMin+1);
            imgData = reshape(imgData(repmat(maskBorder,1,1,1,6)), xMax-xMin+1, yMax-yMin+1, zMax-zMin+1,6);
            
            %visualize pixels
            %         xtTransferImageData(imgData);
            
            maskTemp{index + 1 - startIndex} = byteMask;
            imgTemp{index + 1 - startIndex} = imgData;
            
        end
        
        %store masks, pixels, and other data
        indicesInFile = size(saveFile, 'masks', 1)+1:size(saveFile, 'masks', 1)+endIndex-startIndex+1;
        
%         saveFile.excitations(indicesInFile,1:2) = excitations;
        saveFile.masks(indicesInFile,1) = maskTemp;
        saveFile.imageData(indicesInFile,1) = imgTemp;
%         saveFile.stitchedXYZPositions(indicesInFile,1:3) = [stats(find(strcmp({stats.Name},'Stitched Position X'))).Values(startIndex+1:endIndex+1),...
%             stats(find(strcmp({stats.Name},'Stitched Position Y'))).Values(startIndex+1:endIndex+1),...
%             stats(find(strcmp({stats.Name},'Stitched Position Z'))).Values(startIndex+1:endIndex+1)];
        saveFile.stitchedXYZPositions(indicesInFile,1:3) = [stats(find(strcmp({stats.Name},'Position X'))).Values(startIndex+1:endIndex+1),...
        stats(find(strcmp({stats.Name},'Position Y'))).Values(startIndex+1:endIndex+1),...
        stats(find(strcmp({stats.Name},'Position Z'))).Values(startIndex+1:endIndex+1)];
        
        
        %store surface data in file
        saveFile.vertices(size(saveFile, 'vertices', 1)+1:size(saveFile, 'vertices', 1)+size(surfList.mVertices,1),1:3) = surfList.mVertices;
        saveFile.triangles(size(saveFile, 'triangles', 1)+1:size(saveFile, 'triangles', 1)+size(surfList.mTriangles,1),1:3) = surfList.mTriangles;
        saveFile.normals(size(saveFile, 'normals', 1)+1:size(saveFile, 'normals', 1)+size(surfList.mNormals,1),1:3) =  surfList.mNormals;
        saveFile.timeIndex(size(saveFile, 'timeIndex',1)+1 : size(saveFile, 'timeIndex',1)+length(surfList.mTimeIndexPerSurface),1) = surfList.mTimeIndexPerSurface;
        saveFile.numTriangles(size(saveFile, 'numTriangles',1)+1 : size(saveFile, 'numTriangles',1)+length(surfList.mNumberOfTrianglesPerSurface),1) = surfList.mNumberOfTrianglesPerSurface;
        saveFile.numVertices(size(saveFile, 'numVertices',1)+1 : size(saveFile, 'numVertices',1)+length(surfList.mNumberOfVerticesPerSurface),1) =  surfList.mNumberOfVerticesPerSurface;
        
    end
    %write statistics for all surfaces in batch of time points
    if ~any(strcmp('stats',who(saveFile))) %store first set
        saveFile.stats = stats;
    else %append to existing
        oldStats = saveFile.stats;
        newIds = (0:(length(oldStats(1).Ids) + surface.GetNumberOfSurfaces - 1) )';
        for j = 1:length(stats)
            %modify Ids
            oldStats(j).Ids = newIds;
            %append new values
            oldStats(j).Values = [oldStats(j).Values; stats(j).Values];
        end
        saveFile.stats = oldStats;
    end
    
    saveFile.lastWrittenFrameIndex = startFrame + framesPerLoop - 1;
    
    %delete surfaces now that they're saved
    surface.RemoveAllSurfaces;
    toc
end

%Preprocess statistics for creation of design matrix
statistics = saveFile.stats;
featureNames = {statistics.Name}';
% read Imaris indices
imarisIndices = statistics(1).Ids;
%make design matrix
rawFeatures = cell2mat({statistics.Values});
xPosIdx = find(strcmp(featureNames,'Position X'));
xPosIdy = find(strcmp(featureNames,'Position Y'));
% distanceToBorder = min([rawFeatures(:,xPosIdx), rawFeatures(:,xPosIdy),...
%     max(rawFeatures(:,xPosIdx)) - rawFeatures(:,xPosIdx),  max(rawFeatures(:,xPosIdy)) - rawFeatures(:,xPosIdy) ],[],2);
% inCenter = rawFeatures(:,xPosIdx) > distanceFromEdge & rawFeatures(:,xPosIdx) < (max(rawFeatures(:,xPosIdx)) - distanceFromEdge) &...
%     rawFeatures(:,xPosIdy) > distanceFromEdge & rawFeatures(:,xPosIdy) < (max(rawFeatures(:,xPosIdy)) - distanceFromEdge);
%use only central surfaces
% saveFile.imarisIndices = imarisIndices(inCenter);
saveFile.imarisIndices = imarisIndices;
% excitations = saveFile.excitations;
% saveFile.excitations = excitations(inCenter,:);
% xyzPos = saveFile.stitchedXYZPositions;
% saveFile.stitchedXYZPositions = xyzPos(inCenter,:);
% ti = saveFile.timeIndex;
% saveFile.designMatrixTimeIndices = ti(inCenter);

saveFile.designMatrixTimeIndices = saveFile.timeIndex;

% saveFile.rawFeatures = rawFeatures(inCenter,:);
saveFile.rawFeatures = rawFeatures;
saveFile.rawFeatureNames = featureNames;

end


