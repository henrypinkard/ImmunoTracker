function [] = calcNormalizedCutFeatures(saveFile,...
            features, featureNames, channelOffsets)

maxImagesToCache = 50;

referenceVector = getReferenceVector(saveFile.name);
% imarisIndices = saveFile.imarisIndices;
numSurfaces = length(saveFile.imarisIndices);

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
    cellSize = 11;
    normCutFeatures = cell(length(calcIndices),cellSize);
    for i = 1:length(calcIndices) 
%     parfor i = 1:length(calcIndices)      
        [ totalProjNormedIntesnity, totalProjUnnormedIntesnity, avgProjNormedIntensity,...
            avgProjUnnormedIntensity, roiMeanIntensitySpectrum, roiMeanIntensityMagnitude,...
            corrMatROI, corrMatGlob, roiTotalPixels, roiCentroidOffset, roiCentroid ] =...
            clusterAndCalcFeatures(imageData{i},masks{i}, channelOffsets,referenceVector, saveFile.pixelSizeXY, saveFile.pixelSizeZ);
        %convert roiCentroid to centroid of where we think cell actually is
        centroidIndices = cell2mat(cellfun(@(name) find(strcmp(featureNames,name)),...
                {'Center of Homogeneous Mass X', 'Center of Homogeneous Mass Y',...
                'Center of Homogeneous Mass Z'}, 'UniformOutput', false));
        cellCentroid = roiCentroid + features(calcIndices(i), centroidIndices);    
        normCutFeatures(i,:) = {totalProjNormedIntesnity, totalProjUnnormedIntesnity, avgProjNormedIntensity,...
            avgProjUnnormedIntensity, roiMeanIntensitySpectrum, roiMeanIntensityMagnitude,...
            corrMatROI, corrMatGlob, roiTotalPixels, roiCentroidOffset, cellCentroid};
    end
    saveFile.normCutFeatures(calcIndices,1:cellSize) = normCutFeatures;
    
    %remove indices that were just calculated
    remainingCalcIndices(ismember(remainingCalcIndices,calcIndices)) = [];
    toc
end

end