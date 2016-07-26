clear
load('PreprocessedCMTMRData.mat');

saveFilename = 'CMTMRFeaturesAndLabels.mat';


%Center of homogenous mass XYZ
cohm = features(:,8:10);
%Intensity weighted mass offset
coim = features(:,11:28);
%split by channel
coim1 = coim(:,1:6:end);
coim2 = coim(:,2:6:end);
coim3 = coim(:,3:6:end);
coim4 = coim(:,4:6:end);
coim5 = coim(:,5:6:end);
coim6 = coim(:,6:6:end);

intenistyWeightOffset = zeros(size(features,1),6);
%distance for each channel
intenistyWeightOffset(:,1) = sqrt( sum( (coim1 - cohm).^2, 2) );
intenistyWeightOffset(:,2) = sqrt( sum( (coim2 - cohm).^2, 2) );
intenistyWeightOffset(:,3) = sqrt( sum( (coim3 - cohm).^2, 2) );
intenistyWeightOffset(:,4) = sqrt( sum( (coim4 - cohm).^2, 2) );
intenistyWeightOffset(:,5) = sqrt( sum( (coim5 - cohm).^2, 2) );
intenistyWeightOffset(:,6) = sqrt( sum( (coim6 - cohm).^2, 2) );

featureNames = {featureNames{:} 'Intensity weight offset Ch1' 'Intensity weight offset Ch2'...
    'Intensity weight offset Ch3' 'Intensity weight offset Ch4'...
    'Intensity weight offset Ch5' 'Intensity weight offset Ch6'};
features = [features intenistyWeightOffset];

%pairwise center of image mass differences
for i = 1:6
   for j = 1+1:6 
      %i.j pairwise distance
      featureNames = {featureNames{:} sprintf('Center of Image Mass pairwise distance %i_%i',i,j)}';
      features = [features eval(sprintf('sqrt( sum( (coim%i - coim%i).^2, 2) );',i,j))];
   end
end

%Distance to cortex
load('DistanceTransform.mat')
timeIndices = features(:,96);
posX = features(:,92);
posY = features(:,93);
posZ = features(:,94);
%pixXY is specific to downsampled distance transform
indexX = floor(posX ./ pixXY) + 1;
indexY = floor(posY ./ pixXY) + 1;
indexZ = floor(posZ ./ pixZ) + 1;
transformAtTP = distanceTransforms{timeIndices+1};
distanceToCortex = transformAtTP (sub2ind(size(transformAtTP),indexX,indexY,indexZ));
featureNames = {featureNames{:} 'Distance to cortex'}';
features = [features distanceToCortex];

%interpolation features
pixelSize = 0.363;
indexX = floor(posX ./ pixelSize) + 1;
indexY = floor(posY ./ pixelSize) + 1;
tilePosX = features(:,88);
tilePosY = features(:,89);
tileIndexX = floor(tilePosX ./ pixelSize) + 1;
tileIndexY = floor(tilePosY ./ pixelSize) + 1;
[vertDistBelowSurface, normalAngleWithVertical, normPorjection]...
    = interpolationFeatures([indexX indexY],posZ, [tileIndexX tileIndexY]);

scatter(indexX,indexY, 10,normPorjection,'filled','MarkerEdgeColor','none')
xlabel('Position (um)')
ylabel('Position (um)')
cm = flipud(viridis);
colormap(cm)
colorbar
setFontsAndLines()


featureNames = {featureNames{:} 'Vertical distance below LN surface'...
    'Surface normal angle with vertical', 'Field of view excitation correction'}';
features = [features vertDistBelowSurface normalAngleWithVertical normPorjection];

%getIntensity features
intensityFeatureNames = featureNames(43:84);
intensityFeatureData = features(:,43:84);
%remove from set
featureNames(43:84) = [];
features(:,43:84) = [];


intensityCenters = intensityFeatureData(:,1:6);
intensityMaxs = intensityFeatureData(:,7:12);
intensityMeans = intensityFeatureData(:,13:18);
intensityMedians = intensityFeatureData(:,19:24);
intensityMins = intensityFeatureData(:,25:30);
intensitySTDs = intensityFeatureData(:,31:36);
% intensitySums = intensityFeatureData(:,37:42);   %Intensity sums not needed

channelOffsets = [8 12 16 12 10 8];

%spectral normalization

%subtract offsets
intensityCenters = intensityCenters - repmat(channelOffsets,size(intensityCenters,1),1);
intensityMeans = intensityMeans - repmat(channelOffsets,size(intensityMeans,1),1);
intensityMedians = intensityMedians - repmat(channelOffsets,size(intensityMedians,1),1);
intensityMins  = intensityMins - repmat(channelOffsets,size(intensityMins,1),1);
intensityMaxs  = intensityMaxs - repmat(channelOffsets,size(intensityMaxs,1),1);
%nothing to subtract for standard dev
%Make strictly positive
intensityCenters(intensityCenters < 0) = 0; 
intensityMeans(intensityMeans < 0) = 0; 
intensityMedians(intensityMedians < 0) = 0; 
intensityMins(intensityMins < 0) = 0; 
intensityMaxs(intensityMaxs < 0) = 0; 
%calculate spectral magnitudes
spectralMagnitudeCenter = sqrt(sum(intensityCenters.^2,2));
spectralMagnitudeMean = sqrt(sum(intensityMeans.^2,2));
spectralMagnitudeMedian = sqrt(sum(intensityMedians.^2,2));
spectralMagnitudeMin = sqrt(sum(intensityMins.^2,2));
spectralMagnitudeMax = sqrt(sum(intensityMaxs.^2,2));
spectralMagnitudeSTD = sqrt(sum(intensitySTDs.^2,2));

%remove 0 magnitdudes
spectralMagnitudeCenter(spectralMagnitudeCenter == 0) = 1;
spectralMagnitudeMean(spectralMagnitudeMean == 0) = 1;
spectralMagnitudeMedian(spectralMagnitudeMedian == 0) = 1;
spectralMagnitudeMin(spectralMagnitudeMin == 0) = 1;
spectralMagnitudeMax(spectralMagnitudeMax == 0) = 1;
spectralMagnitudeSTD(spectralMagnitudeSTD == 0) = 1;

%spectral normalization
intensityCenters = intensityCenters ./ repmat(spectralMagnitudeCenter,1,6);
intensityMeans = intensityMeans ./ repmat(spectralMagnitudeMean,1,6);
intensityMedians = intensityMedians ./ repmat(spectralMagnitudeMedian,1,6);
intensityMins = intensityMins ./ repmat(spectralMagnitudeMin,1,6);
intensityMaxs = intensityMaxs ./ repmat(spectralMagnitudeMax,1,6);
intensitySTDs = intensitySTDs ./ repmat(spectralMagnitudeSTD,1,6);

%add magnitudes to set
featureNames = {featureNames{:}, 'Spectral magnitude center','Spectral magnitude mean',...
    'Spectral magnitude median', 'Spectral magnitude min', 'Spectral magnitude max', 'Spectral magnitude stddev'}';
features = [features spectralMagnitudeCenter, spectralMagnitudeMean, spectralMagnitudeMedian,spectralMagnitudeMin,...
    spectralMagnitudeMax, spectralMagnitudeSTD];
%add normalized features to set
newFeatureNames = intensityFeatureNames(1:36); %remove sums
% add to set
featureNames = {featureNames{:}, newFeatureNames{:}}';
features = [features intensityCenters intensityMaxs intensityMeans intensityMedians intensityMins intensitySTDs];

%add Normalized cut features%%%%%%
%%%%%%%%%TODO change to single file for e670
ncDataFilename1 = 'CMTMRTCellAndNonTCellMasksAndImageData';
ncDataFilename2 = 'CMTMRUnlablelledMasksAndImageData';

[imarisIndicesNC, featuresNC, featureNamesNC]  = readNormCutFeatures(ncDataFilename1);
%reshuffle based on imaris indices
[~, ind] = ismember(imarisIndicesNC, imarisIndices);
%index into zeros matrix in case of incoplete set
toAppendFeatures = zeros(size(features,1),length(featureNamesNC));
toAppendFeatures(ind,:) = featuresNC;

[imarisIndicesNC, featuresNC, featureNamesNC]  = readNormCutFeatures(ncDataFilename2);
%reshuffle based on imaris indices
[~, ind] = ismember(imarisIndicesNC, imarisIndices);
%index into zeros matrix in case of incoplete set
toAppendFeatures(ind,:) = featuresNC;

features = [features toAppendFeatures];
featureNames = {featureNames{:}, featureNamesNC{:}}';

% standardize all features
avg = mean(features);
features = features - repmat(avg,size(features,1),1); 
stddev = std(features);
stddev(stddev == 0) = 1;
features = features ./ repmat(stddev, size(features,1),1); 

save(strcat('/Users/henrypinkard/Google Drive/Code/MATLAB/CS289 project/data/',saveFilename),...
    'features','featureNames','imarisIndices','labelledTCell','labelledNotTCell','unlabelled');
