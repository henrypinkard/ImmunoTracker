clear
load('PreprocessedCMTMRData.mat');

saveFilename = 'CMTMRFeaturesAndLabels.mat';
%TODO: read channel ofsets from data?
channelOffsets = [8 12 16 12 10 8];
%TODO: mechanism for reading magellan summary metadata
%TODO: add initial or current surface interpolation into metadata
summaryMD = load('summaryMD');
interpPoints = load('6-6 updated.txt');


%%%%%%%%%%%%  Derived features from differences in COMs and Intensity weighted COMs
[newFeatures, newFeatureNames] = calcCOMFeatures(features, featureNames);
featureNames = {featureNames{:} newFeatureNames{:}}';
features = [features newFeatures]; 

%%%%%%%%%%%%%  interpolation features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[vertDistBelowSurface, normalAngleWithVertical, normPorjection]...
    = calcInterpFeatures(features,featureNames,summaryMD.data, interpPoints);
featureNames = {featureNames{:} 'Vertical distance below LN surface'...
    'Surface normal angle with vertical', 'Field of view excitation correction'}';
features = [features vertDistBelowSurface normalAngleWithVertical normPorjection];

%%%%%%%%%%%%%  Spectral features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[newFeatureNames, newFeatures] = calcSpectralFeatures(features,featureNames, channelOffsets);
featureNames = {featureNames{:}, newFeatureNames{:}}';
features = [features newFeatures];



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
timeIndices = features(:,find(strcmp(featureNames,'Time Index')));

% standardize all features
avg = mean(features);
features = features - repmat(avg,size(features,1),1); 
stddev = std(features);
stddev(stddev == 0) = 1;
features = features ./ repmat(stddev, size(features,1),1); 

save(strcat('/Users/henrypinkard/Google Drive/Research/BIDC/LNImagingProject/data/',saveFilename),...
    'features','featureNames','imarisIndices','labelledTCell','labelledNotTCell','unlabelled','timeIndices');
