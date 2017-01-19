clear
load('Preprocessede670Data.mat');

saveFilename = 'e670FeaturesAndLabels.mat';
%TODO: read channel ofsets from data?
channelOffsets = [8 12 16 12 10 8];
%TODO: mechanism for reading magellan summary metadata
%TODO: add initial or current surface interpolation into metadata
%TODO: remove features that elastic net says are not useful
summaryMD = load('summaryMD');
interpPoints = load('6-6 updated.txt');

ncDataFilename = 'e670MasksAndImageData.mat';



%%%%%%%%%%%%  Derived features from differences in COMs and Intensity weighted COMs
fprintf('Calculating COM features...\n');
[newFeatures, newFeatureNames] = calcCOMFeatures(features, featureNames);
featureNames = {featureNames{:} newFeatureNames{:}}';
features = [features newFeatures]; 

%%%%%%%%%%%%%  interpolation features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Calculating Interpolation features...\n');
[vertDistBelowSurface, normalAngleWithVertical, normPorjection]...
    = calcInterpFeatures(features,featureNames,summaryMD.data, interpPoints);
featureNames = {featureNames{:} 'Vertical distance below LN surface'...
    'Surface normal angle with vertical', 'Field of view excitation correction'}';
features = [features vertDistBelowSurface normalAngleWithVertical normPorjection];

%%%%%%%%%%%%%  Spectral features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Calculating Speactral features...\n');
[newFeatureNames, newFeatures] = calcSpectralFeatures(features,featureNames, channelOffsets);
featureNames = {featureNames{:}, newFeatureNames{:}}';
features = [features newFeatures];

%%%%%%%%%%%%%  Normalized cut features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Calculating Normalized cut features...\n');
[featuresNC, featureNamesNC]  = readNormCutFeatures(ncDataFilename);
features = [features featuresNC];
featureNames = {featureNames{:}, featureNamesNC{:}}';


%TODO: remove features that elasticNet says are not usefule


%%%%%%%%%%% Standardize all features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%store time indices seperately before it is standardized
fprintf('Standardizing and saving...\n');
timeIndices = features(:,find(strcmp(featureNames,'Time Index')));
avg = mean(features);
features = features - repmat(avg,size(features,1),1); 
stddev = std(features);
stddev(stddev == 0) = 1;
features = features ./ repmat(stddev, size(features,1),1); 

save(strcat('/Users/henrypinkard/Google Drive/Research/BIDC/LNImagingProject/data/',saveFilename),...
    'features','featureNames','imarisIndices','timeIndices');
