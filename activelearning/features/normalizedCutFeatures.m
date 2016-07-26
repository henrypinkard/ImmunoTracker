% clear
% 
% %Imaris index 35908 -- in farrred duct
% % Imaris index 20305 -- touching green and farred dendrite
% %Imaris index 156469 -- green red lots of overlap
% %Imaris index 169026 -- green red overlap
% %Imaris index 168870 -- T cell overlap with green-red DC
% 
% file = matfile('CMTMRTCellAndNonTCellMasksAndImageData.mat');
% imarisIndices = file.imarisIndices;
% ii = 35908;
% i = find(imarisIndices{1} == ii);
% img = file.imageData(i,1);
% mask = file.masks(i,1);
% % xtTransferImageData(img);
% clusterAndCalcFeatures(img{1},mask{1});
%% 
clear
maxImagesToCache = 150;

saveFile = matfile('CMTMRUnlablelledMasksAndImageData.mat','Writable',true);

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

tic
while ~isempty(remainingCalcIndices)
    fprintf('remaining calculations: %i\n',length(remainingCalcIndices));
    toc
    calcIndices = remainingCalcIndices(1:min(end,maxImagesToCache));
    %cache pixels and masks for batches at a tiem so as to not overload
    %memory
    imageData = saveFile.imageData(calcIndices,:);
    masks = saveFile.masks(calcIndices,:);
    cellSize = 10;
    normCutFeatures = cell(length(calcIndices),cellSize);
    
    for i = 1:length(calcIndices)
        [ totalProjNormedIntesnity, totalProjUnnormedIntesnity, avgProjNormedIntensity,...
            avgProjUnnormedIntensity, roiMeanIntensitySpectrum, roiMeanIntensityMagnitude,...
            corrMatROI, corrMatGlob, roiTotalPixels, roiCentroidOffset ] =...
            clusterAndCalcFeatures(imageData{i},masks{i});
        normCutFeatures(i,:) = {totalProjNormedIntesnity, totalProjUnnormedIntesnity, avgProjNormedIntensity,...
            avgProjUnnormedIntensity, roiMeanIntensitySpectrum, roiMeanIntensityMagnitude,...
            corrMatROI, corrMatGlob, roiTotalPixels, roiCentroidOffset};
    end
    saveFile.normCutFeatures(calcIndices,1:cellSize) = normCutFeatures;
    
    %remove indices that were just calculated
    remainingCalcIndices(ismember(remainingCalcIndices,calcIndices)) = [];
end
