clear
%%XT Link  forsurface transfer
[ xImarisApp, xPreviewSurface ] = xtSetupSurfaceTransfer(  );
% Load Magellan data
filepath = '/Users/henrypinkard/Desktop/LNData/Left LN_2';
%open magellan instance
javaaddpath('Magellan.jar');
import org.micromanager.plugins.magellan.acq.*;
import org.micromanager.plugins.magellan.misc.*;
mmData = MultiResMultipageTiffStorage(filepath);
%read metadata
summaryMD = mmData.getSummaryMetadata;
imageHeight = MD.getHeight(summaryMD);
imageWidth = MD.getWidth(summaryMD);
xPixelOverlap = MD.getPixelOverlapX(summaryMD);
yPixelOverlap = MD.getPixelOverlapY(summaryMD);

%%%%%%%%%Params
load('Preprocessede670Data.mat')
% %use subset of data
featureIndices = 1:54057;
saveFilename = 'e670MasksAndImageData.mat';
candidateFilename =  '/Users/henrypinkard/Desktop/LNData/e670Candidates.mat';
% featureIndices = 1:length(imarisIndices);
% saveFilename = 'DCMasksAndImageData.mat';
% candidateFilename =  '/Users/henrypinkard/Desktop/LNData/DCCandidates.mat';



%%%%%%%%%%%%%%%%
surfFile = matfile(candidateFilename,'Writable',false);
saveFile = matfile(saveFilename,'Writable',true);
stride = 300;

if ~any(strcmp('imarisIndices',who(saveFile)))
    %start from beginning
    saveFile.masks = {};
    saveFile.imageData = {};
    saveFile.imarisIndices = {imarisIndices(featureIndices)};
    batchStartIndex = 0;
else
    batchStartIndex = floor(size(saveFile,'masks') / stride);
    batchStartIndex = batchStartIndex(1);
end
pixelSizeXY = 0.350819;
pixelSizeZ = 4.5;
%for every surface in candidate set:
tic
for batchIndex = batchStartIndex:length(featureIndices) / stride
    startIndex = batchIndex*stride + 1;
    endIndex = min((batchIndex +1)*stride , length(featureIndices));
    indices = featureIndices(startIndex:endIndex);
    %send it into imaris
    xPreviewSurface.RemoveAllSurfaces; %clear old ones
    func_addsurfacestosurpass(xImarisApp,surfFile,stride,xPreviewSurface,imarisIndices(indices));
    
    maskTemp = cell(stride,1);
    imgTemp = cell(stride,1);
    for surfaceIndex = 1:(endIndex-startIndex + 1)
        %get surface positon
%         position = [features(indices(surfaceIndex), 92)  features(indices(surfaceIndex),93)  features(indices(surfaceIndex),94)];
        position = xPreviewSurface.GetCenterOfMass(surfaceIndex-1);
        %get axis aligned bounding box
        boundingBoxLength = [features(indices(surfaceIndex), 2)  features(indices(surfaceIndex),3)...
            features(indices(surfaceIndex),4)];
        %make it extra big to make sure all is captured
        boundingBoxLength = 1*boundingBoxLength;
        
        %get its mask from imaris
        topLeft = position -  boundingBoxLength / 2;
        bottomRight = position + boundingBoxLength / 2;
        topLeftPix = int32(floor([topLeft(1:2) ./ pixelSizeXY topLeft(3) ./ pixelSizeZ]));
        bottomRightPix = int32(floor([bottomRight(1:2) ./ pixelSizeXY bottomRight(3) ./ pixelSizeZ]));
        pixelResolution = bottomRightPix - topLeftPix + 1;
        mask = xPreviewSurface.GetSingleMask(surfaceIndex-1, topLeft(1), topLeft(2), topLeft(3), bottomRight(1), bottomRight(2),...
            bottomRight(3), pixelResolution(1), pixelResolution(2), pixelResolution(3));
        byteMask = uint8(squeeze(mask.GetDataBytes));
        
        timeIndex = xPreviewSurface.GetTimeIndex(surfaceIndex-1);
        imgData = readRawMagellan( mmData, byteMask, topLeftPix - int32([xPixelOverlap/2, yPixelOverlap/2, 0]), timeIndex );
        
        %trim to border of mask, but leave an extra pixel in xy for
        %filtering
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
        
        maskTemp{surfaceIndex} = byteMask;
        imgTemp{surfaceIndex} = imgData;

        fprintf('surface index %i\n',surfaceIndex)
    end
    saveFile.masks(startIndex:startIndex+stride-1,1) = maskTemp;
    saveFile.imageData(startIndex:startIndex+stride-1,1) = imgTemp;
    fprintf('batch %i of %i complete\n',batchIndex,floor(length(featureIndices) / stride))
    toc
end

