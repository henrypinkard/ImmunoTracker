% clear
% 
% %Imaris index 35908 -- in farrred duct
% % Imaris index 20305 -- touching green and farred dendrite
% %Imaris index 156469 -- green red lots of overlap
% %Imaris index 169026 -- green red overlap
% %Imaris index 168870 -- T cell overlap with green-red DC
% 
% referenceVector = [0.1024    0.0700    0.0953    0.0691    0.9660    0.1939]; %CMTMR
% channelOffsets = [8 12 16 12 10 8];
% file = matfile('CMTMRTCellAndNonTCellMasksAndImageData.mat');
% imarisIndices = file.imarisIndices;
% ii = 168870;
% i = find(imarisIndices{1} == ii);
% img = file.imageData(i,1);
% mask = file.masks(i,1);
% % xtTransferImageData(img);
% clusterAndCalcFeatures(img{1},mask{1},channelOffsets,referenceVector);
%% 
clear
maxImagesToCache = 50;

saveFile = matfile('e670MasksAndImageData.mat','Writable',true);

channelOffsets = [8 12 16 12 10 8];
% referenceVector = [0.1024    0.0700    0.0953    0.0691    0.9660    0.1939]; %CMTMR
referenceVector = [0.0877    0.0642    0.0684    0.0049    0.1772    0.9757]; %e670
% referenceVector =   [0.0701    0.0660    0.9262    0.3365    0.1170    0.0777]; %green DC
% referenceVector =   [  0.0506    0.0347    0.8640    0.3520    0.1245    0.3322]; %green pink DC
% referenceVector =   [ 0.0844    0.1937    0.7024    0.5647    0.3643 0.1018];   %autofluor DC
% referenceVector = [0.0426    0.0297    0.6433    0.3601    0.6675 0.0909]; %green red DC

imarisIndices = saveFile.imarisIndices(1,1);
numSurfaces = length(imarisIndices{1});

if ~any(strcmp('normCutFeatures',who(saveFile)))
    %create cell to store features if needed
    saveFile.normCutFeatures = cell(1);
    remainingCalcIndices = 1:numSurfaces;
else
    remainingCalcIndices = size(saveFile,'normCutFeatures',1) + 1:numSurfaces;
end

%figure out which ones are empty

while ~isempty(remainingCalcIndices)
    tic
    fprintf('remaining calculations: %i\n',length(remainingCalcIndices));
    fprintf('\n\n');
    calcIndices = remainingCalcIndices(1:min(end,maxImagesToCache));
    %cache pixels and masks for batches at a tiem so as to not overload
    %memory
    imageData = saveFile.imageData(calcIndices,:);
    masks = saveFile.masks(calcIndices,:);
    cellSize = 10;
    normCutFeatures = cell(length(calcIndices),cellSize);
    parfor i = 1:length(calcIndices)      
        [ totalProjNormedIntesnity, totalProjUnnormedIntesnity, avgProjNormedIntensity,...
            avgProjUnnormedIntensity, roiMeanIntensitySpectrum, roiMeanIntensityMagnitude,...
            corrMatROI, corrMatGlob, roiTotalPixels, roiCentroidOffset ] =...
            clusterAndCalcFeatures(imageData{i},masks{i}, channelOffsets,referenceVector);
        normCutFeatures(i,:) = {totalProjNormedIntesnity, totalProjUnnormedIntesnity, avgProjNormedIntensity,...
            avgProjUnnormedIntensity, roiMeanIntensitySpectrum, roiMeanIntensityMagnitude,...
            corrMatROI, corrMatGlob, roiTotalPixels, roiCentroidOffset};
    end
    saveFile.normCutFeatures(calcIndices,1:cellSize) = normCutFeatures;
    
    %remove indices that were just calculated
    remainingCalcIndices(ismember(remainingCalcIndices,calcIndices)) = [];
    toc
end
