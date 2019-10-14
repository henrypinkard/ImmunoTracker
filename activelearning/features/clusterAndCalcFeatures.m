function [ totalProjNormedIntesnity, totalProjUnnormedIntesnity, avgProjNormedIntensity,...
    avgProjUnnormedIntensity, roiMeanIntensitySpectrum, roiMeanIntensityMagnitude,...
    corrMatROI, corrMatGlob, roiTotalPixels, roiCentroidOffset, roiCentroid] = clusterAndCalcFeatures( pixels, mask,...
    channelOffsets, referenceVector, pixelSizeXY, pixelSizeZ)
% totalProjNormedIntesnity -- sum of all projected normalized pixels in ROI
% totalProjUnnormedIntesnity -- sum of all projected unnormalized pixels in ROI
% avgProjNormedIntensity -- average projected normalized intenisty in ROI
% avgProjUnnormedIntensity -- average projected unnormalized intensity
% roiMeanIntensitySpectrum -- 1x6 vector of spectrally normalized mean intensities
% roiMeanIntensityMagnitude -- magnitude of mean intensity
% corrMatROI -- Correlation matrix for unnormalized pixels in ROI
% corrMatGlob -- correlation matrix for all pixels masked by surface
% roiCentroid -- 1x3 xyz coordinates of ROI centroid offset from whole mask ROI

%pixels are x,y,z,c

%TODO: which of these need to be changed with new data with new piel sizes
%etc

maxEclideanDistance = 10;
numSlices2Use = 6; %window size of slices used for ROI calculation
pixelsPerCluster = 60; %approximate size of T cell

alpha = 0.7; %spatial distance weight
beta = 1; %spectral distance weight
gamma = 0; %intenisty distance weight


channelOffsets = uint8(channelOffsets);
channelOffsets = reshape(channelOffsets,1,1,1,6);
%background subtract pixels
pixels = double(pixels) - double(repmat(channelOffsets,size(pixels,1),size(pixels,2),size(pixels,3),1));

%remove border to limit filtering artifacts
% pixels = pixels(2:end-1,2:end-1,:,:);
% mask = mask(2:end-1,2:end-1,:);
%smooth pixels
pixels = imgaussfilt(pixels,1);
%remove slices with no masked pixels
validZs = find(squeeze(sum(sum(mask,1),2)));
pixels = pixels(:,:,validZs,:);
mask = mask(:,:,validZs);

numSlices = size(pixels,3);
numPixels = cell(numSlices,1);
normProjectedIntenistySum = cell(numSlices,1);
normProjectedIntenistyAvg = cell(numSlices,1);
unnormProjectedIntenistySum = cell(numSlices,1);
unnormProjectedIntenistyAvg = cell(numSlices,1);

spectralMag = sqrt(sum(double(pixels).^2,4));
normalizationFactor = repmat(spectralMag,1,1,1,6);
normalizationFactor(normalizationFactor == 0) = 1;
normalizedPix = double(pixels) ./ normalizationFactor;
clusterMasks = cell(numSlices,1);

sliceMasks = arrayfun(@(slcIdx) mask(:,:,slcIdx),1:numSlices,'UniformOutput', false);
sliceMaskIndices = cellfun(@find, sliceMasks,'UniformOutput',false);
for sliceIndex = 1:numSlices
    %% compute distances
    sliceMask = sliceMasks{sliceIndex};
    maskIndices = sliceMaskIndices{sliceIndex};
    maskSize = size(sliceMask);
    %each pixel is a vertex
    nPixels = sum(sliceMask(:));
    %compute number of clusters based on number of pixels
    numClusters = max(1,ceil(nPixels ./ pixelsPerCluster));
    [pixelCoordsX, pixelCoordsY, pixelCoordsZ] = arrayfun(@(index) ind2sub(maskSize,maskIndices(index)), 1:nPixels );
    pixelCoordsMicron = [pixelSizeXY*pixelCoordsX' pixelSizeXY*pixelCoordsY' pixelSizeZ*pixelCoordsZ'];
    %compute distance matrices
    rawSpaceDist = squareform(pdist(pixelCoordsMicron,'squaredeuclidean'));
    rawSpaceDist(rawSpaceDist > maxEclideanDistance^2) = Inf;
    pixVec = reshape(pixels(:,:,sliceIndex,:),[],6);
    normalizedPixVec = reshape(normalizedPix(:,:,sliceIndex,:),[],6);
    normalizedMaksedPixVec = normalizedPixVec(maskIndices,:);
    [i,j] = ind2sub(size(rawSpaceDist),find(~isinf(rawSpaceDist)));
    ij = [i, j];
    ij = ij(i > j,:);
    rawSpecDist = zeros(nPixels);
    rawIntensityDist = zeros(nPixels);
    if ~isempty(ij) %prevents error when theres only one pixel
        specDistVec = 1 - dot(normalizedMaksedPixVec(ij(:,1),:),normalizedMaksedPixVec(ij(:,2),:),2);
        rawSpecDist( sub2ind(size(rawSpecDist),ij(:,1), ij(:,2)) ) = specDistVec;      
        sliceSpectralMag = spectralMag(:,:,sliceIndex);
        spectralMagVec = sliceSpectralMag(:);
        maskSpectralMags = spectralMagVec(maskIndices,:);
        intensityDistVec =  abs(maskSpectralMags(ij(:,1)) - maskSpectralMags(ij(:,2)));
        rawIntensityDist( sub2ind(size(rawIntensityDist),ij(:,1), ij(:,2)) ) = intensityDistVec;
    end
    %mirror
    rawSpecDist = rawSpecDist + rawSpecDist';
    rawIntensityDist = rawIntensityDist + rawIntensityDist';
    
    %% Create composite distance matrix
    %normalize by maximum distance before truncation
    spaceDist = rawSpaceDist ./ maxEclideanDistance.^2;
    intensityDist = rawIntensityDist ./ max(1,max(rawIntensityDist(:)));
    intensityDist = sqrt(intensityDist); %equalize histogram
    specDist = rawSpecDist;
    specDist(rawSpecDist < 0) = 0;
%     specDist = sqrt(specDist);  %equalize histogram

    %transform to equalize histograms
    % figure(1)
    % hist(spaceDist( ~isinf(spaceDist) & spaceDist ~=0),30)
    % figure(2)
    % hist((intensityDist( ~isinf(spaceDist) & spaceDist ~=0)),100)
    % figure(3)
    % hist((specDist( ~isinf(spaceDist) & spaceDist ~=0)),100)
    
    
    distMat = spaceDist*alpha + specDist*beta + intensityDist*gamma;
    % distMat =  (specDist*beta + intensityDist*gamma);
    % distMat = specDist;
    
    %set diagonals to infinte dist
    distMat(1:size(distMat,1)+1:size(distMat,1)^2) = inf;
    Adjacency = exp(-distMat );
    
    %% Normalized cut
    
    clusterImg = zeros(size(sliceMask));
    %initialize with all masked pixels bekonging to first cluster
    clusterImg(maskIndices) = 1;
    clusterMasks{sliceIndex,1 } = ones(size(Adjacency,1),1);
    %if not enough pixels, just leave all belonging to same
    %cluster..shouldn't affect output too much
    if size(Adjacency,1) >= numClusters
        C = [];
        count = 0;
        while isempty(C)
            try
                C = SpectralClustering(Adjacency,numClusters,3);
            catch
                count = count+1;
                fprintf('Kmeans failed, trying again count = %i\n',count);
            end
        end
        [a, b] = ind2sub(size(C),find(C));
        [~, I] = sort(a);
        clusterIndex = b(I);
        clusterImg(maskIndices) = clusterIndex;
        
        for c = 1:numClusters
            %mask for each cluster within masked pixels at slice
            clusterMasks{ sliceIndex, c } = clusterIndex == c;
        end
    end

    
    %% Compute feature
    %find the total normalized Red in each cluster
    normalizedPixInSlice = normalizedPix(:,:,sliceIndex,:);
    pixInSlice = pixels(:,:,sliceIndex,:);
    normalizedPixByCluster = cellfun(@(clusterIndex) reshape(normalizedPixInSlice(repmat(clusterIndex == clusterImg,1,1,1,6)),[],6),...
        mat2cell([1:numClusters],1,ones(numClusters,1)),'UniformOutput',false);
    pixByCluster = cellfun(@(clusterIndex) reshape(pixInSlice(repmat(clusterIndex == clusterImg,1,1,1,6)),[],6),...
        mat2cell([1:numClusters],1,ones(numClusters,1)),'UniformOutput',false);
    
    unnormalizedProjIntenstySum = cellfun(@(mat) sum(dot(mat,repmat(referenceVector,size(mat,1),1),2)),pixByCluster);
    normalizedProjectedIntenistySum = cellfun(@(mat) sum(dot(mat,repmat(referenceVector,size(mat,1),1),2)),normalizedPixByCluster);
    numPixelsByCluster = cellfun(@(x) size(x,1), normalizedPixByCluster);
    
    numPixels{sliceIndex} = numPixelsByCluster;
    normProjectedIntenistySum{sliceIndex} = normalizedProjectedIntenistySum;
    normProjectedIntenistyAvg{sliceIndex} = normalizedProjectedIntenistySum ./ numPixelsByCluster;
    unnormProjectedIntenistySum{sliceIndex} = unnormalizedProjIntenstySum;
    unnormProjectedIntenistyAvg{sliceIndex} = unnormalizedProjIntenstySum ./ numPixelsByCluster;

%     % Visualize
%     figure(1)
%     imshow(imfuse(pixels(:,:,sliceIndex,3),pixels(:,:,sliceIndex,5)),[])
%     %green image
%     figure(2)
%     imshow(pixels(:,:,sliceIndex,3),[])
%     colormap([zeros(256,1) linspace(0,1,256)' zeros(256,1)])
%     %red image
%     figure(3)
%     redImg = pixels(:,:,sliceIndex,5);
%     imshow(redImg,[min(redImg(:)) max(redImg(:))])
%     colormap([linspace(0,1,256)' zeros(256,1) linspace(0,1,256)'])
%     %show mask
%     figure(4)
%     imshow(clusterImg > 0)
%     colormap([0 0 0; 1 0 0])
% 
%     
%     %show segmented regions
%     % contrast modification so regions are distinct form background
%     clusterImgForDisplay = clusterImg;
%     clusterImgForDisplay(clusterImg ~= 0) = clusterImgForDisplay(clusterImg ~= 0);
%     figure(5)
%     imshow(clusterImgForDisplay,[])
%     %Hightlight sorted regions
%     
%     colormap ([inferno; viridis])
%     figure(6)
%     clusterImg2 = zeros(size(clusterImg));
%     avg = normalizedProjectedIntenistySum ./ numPixelsByCluster;
%     [asd,I] = sort(avg,'descend');
%     clusterImg2(clusterImg == I(1)) = 100;
%     imshow(clusterImg2,[])
%     colormap([0 0 0; 1 0 0])
%     if sliceIndex == 7 || sliceIndex == 8
%        bleh = 5; 
%     end
end

%Sort by average projected intensity
[normProjectedIntenistyAvgSorted, I]  = cellfun(@(x) sort(x,'descend'), normProjectedIntenistyAvg, 'UniformOutput', false);
normProjectedIntenistySumSorted = cellfun(@(np,ind) np(I{ind}), normProjectedIntenistySum', num2cell(1:length(I)),'UniformOutput', false);
unnormProjectedIntenistySumSorted = cellfun(@(np,ind) np(I{ind}), unnormProjectedIntenistySum', num2cell(1:length(I)),'UniformOutput', false);
unnormProjectedIntenistyAvgSorted = cellfun(@(np,ind) np(I{ind}), unnormProjectedIntenistyAvg', num2cell(1:length(I)),'UniformOutput', false);
numPixelsSorted = cellfun(@(np,ind) np(I{ind}), numPixels', num2cell(1:length(I)),'UniformOutput', false);

if numSlices2Use > numSlices
    slices2Use = 1:numSlices;
else
    %find contiguous window where the most relevant intensity is contained
    roiAvgIntensity = cellfun(@(x) x(1),normProjectedIntenistyAvgSorted);
    %filter the averge projected intenisty at each slice to figure out which
    %window to use for calculations
    movingSum = @(vec) arrayfun(@(start) sum(vec(start:start+numSlices2Use-1)), 1:length(vec)-numSlices2Use+1);
    %find the number of slices that is T cell sized and contains the most
    %projected avg intensity
    [~, useSliceStart] = max(movingSum(roiAvgIntensity));
    slices2Use = useSliceStart:useSliceStart + numSlices2Use - 1;
end

extractROIs = @(cellOfSortedClusters) cellfun(@(x) x(1),cellOfSortedClusters);
%total intensity, normalized and unnormalized
roiProjectedIntensitySumNormed = extractROIs(normProjectedIntenistySumSorted); %take cluster of interest
totalProjNormedIntesnity = sum(roiProjectedIntensitySumNormed(slices2Use)); %sum over all slices
roiProjectedIntensitySumUnnormed = extractROIs(unnormProjectedIntenistySumSorted);
totalProjUnnormedIntesnity = sum(roiProjectedIntensitySumUnnormed(slices2Use));

%average intensity, normalized and unnormalized
%take average for unnormalized intensity, but normalized intenisties must
%be weighted by number of pixels
numPixelsInROIs = extractROIs(numPixelsSorted);
roiProjectedIntensityAvgNormed = extractROIs(normProjectedIntenistyAvgSorted);
avgProjNormedIntensity = sum(roiProjectedIntensityAvgNormed(slices2Use) .*...
    numPixelsInROIs(slices2Use)') ./ sum(numPixelsInROIs(slices2Use));
roiProjectedIntensityAvgUnnormed = extractROIs(unnormProjectedIntenistyAvgSorted);
avgProjUnnormedIntensity = mean(roiProjectedIntensityAvgUnnormed(slices2Use));

%total number of pixels
roiTotalPixels = sum(numPixelsInROIs(slices2Use));

%compute correlation matrix for all pixel values
pixVec = reshape(double(pixels),[],6);
maksedPixVec = pixVec(find(mask),:);
corrMatGlob = corrcov(cov(maksedPixVec));
corrMatGlob(isnan(corrMatGlob)) = 0; %replace undefined entries

%ROI correlation matrix
useClusterInd = cellfun(@(x) x(1),I);
singleClusterMasks = arrayfun(@(sliceI, sortI) clusterMasks(sliceI,sortI),slices2Use',useClusterInd(slices2Use));
pixelClusterMasks = cellfun(@(clusterM, sliceM) sliceM(clusterM), singleClusterMasks,sliceMaskIndices(slices2Use)','UniformOutput',false);
%get pixels corresponding to each cluster
pixelXYFlat = reshape(pixels,[],size(pixels,3),size(pixels,4));
pixelsInROIs = cell2mat(cellfun(@(mask,z) reshape(squeeze(pixelXYFlat(mask,z,:)),[],6),pixelClusterMasks,num2cell(slices2Use'),'UniformOutput',false));
corrMatROI = corrcov(cov(pixelsInROIs));
corrMatROI(isnan(corrMatROI)) = 0; %replace undefined entries

%spectrally normalized mean intensity for all pixels in ROIs
meanIntensities = mean(pixelsInROIs);
roiMeanIntensityMagnitude = norm(meanIntensities);
roiMeanIntensitySpectrum = meanIntensities ./ roiMeanIntensityMagnitude;

%coordinates of roi Centroid
zCoords = repmat(reshape(pixelSizeZ *(0:size(mask,3)-1),1,1,[]),size(mask,1),size(mask,2));
xCoords = repmat(reshape(pixelSizeXY *(0:size(mask,1)-1),[],1,1),1,size(mask,2),size(mask,3));
yCoords = repmat(reshape(pixelSizeXY *(0:size(mask,2)-1),1,[],1),size(mask,1),1,size(mask,3));
maskCentroid = [mean(xCoords(logical(mask)))  mean(yCoords(logical(mask)))  mean(zCoords(logical(mask)))];
%find centroid of ROI
flattenXY = @(mat) reshape(mat,[],size(mat,3));
xFlat = flattenXY(xCoords);
yFlat = flattenXY(yCoords);
zFlat = flattenXY(zCoords);
roiXCoords = cell2mat(cellfun(@(mask,z) squeeze(xFlat(mask,z)),pixelClusterMasks,num2cell(slices2Use'),'UniformOutput',false));
roiYCoords = cell2mat(cellfun(@(mask,z) squeeze(yFlat(mask,z)),pixelClusterMasks,num2cell(slices2Use'),'UniformOutput',false));
roiZCoords = cell2mat(cellfun(@(mask,z) squeeze(zFlat(mask,z)),pixelClusterMasks,num2cell(slices2Use'),'UniformOutput',false));
roiCentroid = [mean(roiXCoords)  mean(roiYCoords)  mean(roiZCoords)];

roiCentroidOffset = roiCentroid - maskCentroid;
end


