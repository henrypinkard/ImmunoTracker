function [] = createSurfaces(surfName,...
    channelIndex, smoothFilter, localContrast, backgroundSub, seedDiam, quality, minNumVoxels)
%CREATESURFACES
% Open a Imaricumpiler "Positions as time" file in Imaris
% this function will prompt user to open  a Micro-Magellan dataset
% to read calibrtion and posiotion metadata
% channelIndex is 0 based, but shows up in Imaris as 1 based
%creates and saves surfaces and statistics, reads image data and masks

%rank filtered low res (for NN training data
%GFP
%     channelIndex = 2;
%     smoothFilter = 1.5;
%     localContrast = 8;
%     backgroundSub = 10;
%     seedDiam = 7;
%   quality = 1
%number of voxels = 150

%e670
%     channelIndex = 5;
%     smoothFilter = 1.5;
%     localContrast = 10;
%     backgroundSub = 10;
%     seedDiam = 12;
%   quality = 1
%number of voxels = 150

%non rank filtered
%3x3x3 median filter
%e670 ltmphocytes on lo res gen3
%     channelIndex = 5;
%     smoothFilter = 1.5;
%     localContrast = 20;
%     backgroundSub = 1.8;
%     seedDiam = 15;
%   quality = 0.1
%number of voxels = 100

%GFP DCs on high res Gen3
%     channelIndex = 2;
%     smoothFilter = 1.5;
%     localContrast = 25;
%     backgroundSub = 2.5;
%     seedDiam = 21;
%   quality = 1
%number of voxels = 150

%CMTMR/e670 T cells on high res Gen3
%     channelIndex = 4/5;
%     smoothFilter = 1.5;
%     localContrast = 20;
%     backgroundSub = 2;
%     seedDiam = 15;
%   quality = 0.5
%number of voxels = 120

%change to folder above
cd(fileparts(mfilename('fullpath')))

%Parameters that can be tuned to optimize performance
batchSize = 100;
framesPerLoop = 2; %number of frames for which surfaces are created with each loop

%add java libraries to dynamic path
javaaddpath('./ImarisLib.jar');

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
saveName = strcat(imsFilePathAndName,'_',char(surfName),'.mat');
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
    saveFile.name = surfName;
    saveFile.surfInterpPoints = {};
%     saveFile.summaryMD = summaryMD;
    %get channel offsets. this is stochastic so do it a few times and median
    offsets = zeros(6, 1);
    for c  = 0:5
        offsets(c + 1) = median(ds.GetDataVolumeAs1DArrayBytes(c, 0));
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
        excitations = zeros(endIndex-startIndex+1,2);
        %iterate through each surface to get its mask and image data
        for index = startIndex:endIndex
            %get axis aligned bounding box
            boundingBoxSize = [stats(find(strcmp({stats.Name},'BoundingBoxAA Length X'))).Values(index + 1),...
                stats(find(strcmp({stats.Name},'BoundingBoxAA Length Y'))).Values(index + 1),...
                stats(find(strcmp({stats.Name},'BoundingBoxAA Length Z'))).Values(index + 1)];
            positionUnstitched = surface.GetCenterOfMass(index);            
            
            %Mask is in unstitched coordinates, image data is in stitched coordinates
            topLeft = positionUnstitched -  boundingBoxSize / 2;
            bottomRight = positionUnstitched + boundingBoxSize / 2;
%             topLeftStitched = positionStitched -  boundingBoxSize / 2;
            pixelResolution = int32(floor((bottomRight - topLeft) ./ [pixelSizeXY pixelSizeXY pixelSizeZ]));
            topLeftPix = int32(floor([topLeft(1:2) ./ pixelSizeXY topLeft(3) ./ pixelSizeZ]));
            mask = surface.GetSingleMask(index, topLeft(1), topLeft(2), topLeft(3), bottomRight(1), bottomRight(2),...
                bottomRight(3), pixelResolution(1), pixelResolution(2), pixelResolution(3));
            byteMask = uint8(squeeze(mask.GetDataBytes));
            timeIndex = surface.GetTimeIndex(index);
            imgData = zeros(pixelResolution(1), pixelResolution(2), pixelResolution(3), 6);
            for c = 0:5
                imgData(:,:,:,c+1) = ds.GetDataSubVolumeBytes(topLeftPix(1), topLeftPix(2), topLeftPix(3), c, timeIndex,...
                pixelResolution(1), pixelResolution(2), pixelResolution(3));
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
            
            %get surface interpolation and laser excitation
%             try
%                 mdString = char(md.toString);                
%                 mdString(mdString == ' ') = '';
%                 mdString(mdString == '#') = '';
%                 mdString(mdString == ')') = '';
%                 mdString(mdString == '(') = '';
%                 md = JSON.parse(mdString);
%                 laser1Power = cellfun(@str2num,strsplit(md.TeensySLM1_SLM_Pattern,'-'),'UniformOutput',0);
%                 laser2Power = cellfun(@str2num,strsplit(md.TeensySLM2_SLM_Pattern,'-'),'UniformOutput',0);
%                 laser1Power = cell2mat(reshape(laser1Power(1:end-1),16,16));
%                 laser2Power = cell2mat(reshape(laser2Power(1:end-1),16,16));
%                 %bilinear interpolation
%                 indices = (positionUnstitched(1:2) ./ fovSize) * 15;
%                 fractional = mod(indices,1.0);
%                 lowerIndices = indices - fractional + 1;
%                 lowerIndexWeight = 1 - fractional;
%                 blinterp = @(grid, lowerInd, lowerWeight)...
%                     grid(lowerInd(1), lowerInd(2)) * lowerWeight(1) * lowerWeight(2) +...
%                     grid(lowerInd(1) + 1, lowerInd(2)) * (1- lowerWeight(1)) * lowerWeight(2) +...
%                     grid(lowerInd(1), lowerInd(2) + 1) * lowerWeight(1) * (1 - lowerWeight(2)) +...
%                     grid(lowerInd(1) + 1, lowerInd(2) + 1) * (1- lowerWeight(1)) * (1 - lowerWeight(2));                              
%                 excitations(index + 1 - startIndex,:) = [blinterp(laser1Power, lowerIndices, lowerIndexWeight) blinterp(laser2Power, lowerIndices, lowerIndexWeight)];                
%                 %store one copy of surface interpoaltion per a time point
%                 if size(saveFile, 'surfInterpPoints', 1) < timeIndex + 1 || isempty(saveFile.surfInterpPoints(timeIndex+1,1))
%                     interpPoints = md.DistanceFromFixedSurfacePoints;
%                     saveFile.surfInterpPoints = cell2mat(cellfun(@(entry) cellfun(@str2num, strsplit(entry,'_')),interpPoints,'UniformOutput',0)');
%                 end
%             catch
%                 excitations(index + 1 - startIndex,:) = nan;
%             end
        end
        
        %store masks, pixels, and other data
        indicesInFile = size(saveFile, 'masks', 1)+1:size(saveFile, 'masks', 1)+endIndex-startIndex+1;
        
%         saveFile.excitations(indicesInFile,1:2) = excitations;
        saveFile.masks(indicesInFile,1) = maskTemp;
        saveFile.imageData(indicesInFile,1) = imgTemp;
%         saveFile.stitchedXYZPositions(indicesInFile,1:3) = [stats(find(strcmp({stats.Name},'Stitched Position X'))).Values(startIndex+1:endIndex+1),...
%             stats(find(strcmp({stats.Name},'Stitched Position Y'))).Values(startIndex+1:endIndex+1),...
%             stats(find(strcmp({stats.Name},'Stitched Position Z'))).Values(startIndex+1:endIndex+1)];
        
        
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

% saveFile.rawFeatures = rawFeatures(inCenter,:);
saveFile.rawFeatures = rawFeatures;
saveFile.rawFeatureNames = featureNames;

% Copy to backup directory on different drive
% if (any(imsFilePathAndName == '\'))
%     dirs = strsplit(imsFilePathAndName,'\');
% else
%     dirs = strsplit(imsFilePathAndName,'/');
% end
% copyfile(saveName, strcat(backupDirectory,dirs{end-1},'_',dirs{end},'_',char(surfName),'.mat') );

% mmData.close;
% clear mmData;

%     function [newStats] = modify_stats_for_stitched_view(stats)
%         newStats = stats;
%         pxIdx = find(ismember({stats.Name},'Position X'));
%         pyIdx = find(ismember({stats.Name},'Position Y'));
%         pzIdx = find(ismember({stats.Name},'Position Z'));
%         tIdxIdx = find(ismember({stats.Name},'Time Index'));
%         
%         %rename original position stats
%         % newStats(pxIdx).Name = 'Tile Position X';
%         % newStats(pyIdx).Name = 'Tile Position Y';
%         % newStats(pzIdx).Name = 'Tile Position Z';
%         
%         timeIndex = newStats(tIdxIdx).Values;
%         %time index stat is one based, so subtract one
%         stitchedTimeIndex = mod(timeIndex - 1,numTimePoints);
%         %add actual time index in
%         newStats(tIdxIdx).Values = stitchedTimeIndex;
%         
%         posIndices = floor(double(timeIndex - 1) ./ numTimePoints);
%         posIndicesCell = num2cell(posIndices);
%         
%         rows = cellfun(@(index) posList.get(index).getInt('GridRowIndex'),posIndicesCell);
%         cols = cellfun(@(index) posList.get(index).getInt('GridColumnIndex'),posIndicesCell);
%         %calculate offset to translate from individual field of view to proper
%         %position in stitched image
%         translation = [(cols * (imageWidth - xPixelOverlap)) * pixelSizeXY, (rows * (imageHeight - yPixelOverlap)) * pixelSizeXY];
%         
%         singlestruct = @(name, values) struct('Ids',newStats(1).Ids,'Name',name,'Values',values,'Units','');
%         
%         newStats(length(newStats) + 1) = singlestruct('Stitched Position X',newStats(pxIdx).Values + translation(:,1));
%         newStats(length(newStats) + 1) = singlestruct('Stitched Position Y',newStats(pyIdx).Values + translation(:,2));
%         newStats(length(newStats) + 1) = singlestruct('Stitched Position Z',newStats(pzIdx).Values );
%         
%         %sort into alphabetical order
%         [~, statOrder] = sort({newStats.Name});
%         newStats = newStats(statOrder);
%     end
end

% function [ pixelData, metadata ] = readRawMagellan( magellanDataset, surfaceMask, offset, timeIndex )
% %return metadata from middle requested slice
% %offset is in in 0 indexed pixels
% %timeIndexStarts with 0
% unsign = @(arr) uint8(bitset(arr,8,0)) + uint8(128*(arr < 0));
% 
% metadata = [];
% %Read raw pixels from magellan
% %take stiched pixels instead of decoding position indices
% pixelData = zeros(size(surfaceMask,1),size(surfaceMask,2),size(surfaceMask,3),6, 'uint8');
% for channel = 0:5
%     for relativeSlice = 0:size(surfaceMask,3)-1
%         slice = offset(3) + relativeSlice;
%         ti = magellanDataset.getImageForDisplay(channel, slice, timeIndex, 0, offset(1), offset(2),...
%             size(pixelData,1), size(pixelData,2));
%         pixels = reshape(unsign(ti.pix),size(pixelData,1),size(pixelData,2));
%         if (isempty(metadata) && floor(size(surfaceMask,3)/2) == relativeSlice )
%             metadata = ti.tags;
%         end
%         pixelData(:,:,relativeSlice+1,channel+1) = pixels;
%     end
% end

% end


% function [newVertices, newTimeIndex] = modify_surfaces_for_stitched_view(vertices, timeIndex, numVertices, posList, imageWidth, imageHeight,...
%     xPixelOverlap, yPixelOverlap, pixelSize, numTimePoints)
% 
% newVertices = zeros(size(vertices));
% newTimeIndex = zeros(size(timeIndex));
% 
% %go through each surface and add it to the new surfaces as appropriate
% posIndices = floor(double(timeIndex) ./ numTimePoints);
% 
% surfCount = 0;
% while surfCount < length(timeIndex)
%     %process in batches with same positon index (which means same offset in
%     %larger sitched image)
%     numInBatch = find(posIndices(1 + surfCount:end) ~= posIndices(1 + surfCount),1) - 1;
%     if (isempty(numInBatch))
%         numInBatch = length(posIndices) - surfCount; %last batch
%     end
%     verticesInBatch = sum(numVertices(1+surfCount:surfCount+numInBatch));
%     firstVertex = 1 + sum(numVertices(1:surfCount));
%     
%     
%     %offsets for spots in the larger stitched image
%     row = posList.get(posIndices(1 + surfCount)).getInt('GridRowIndex');
%     col = posList.get(posIndices(1 + surfCount)).getInt('GridColumnIndex');
%     %calculate offset to translate from individual field of view to proper
%     %position in stitched image
%     offset = [(col * (imageWidth - xPixelOverlap)) * pixelSize, (row * (imageHeight - yPixelOverlap)) * pixelSize, 0];
%     
%     %modify vertices and time
%     newVertices(firstVertex:firstVertex+verticesInBatch - 1,:) = vertices(firstVertex:firstVertex+verticesInBatch - 1,:) + repmat(offset,verticesInBatch,1);
%     newTimeIndex(1+surfCount:surfCount+numInBatch) = mod(timeIndex(1+surfCount:surfCount+numInBatch),numTimePoints);
%     
%     surfCount = surfCount + numInBatch;
% end
% 
% end

