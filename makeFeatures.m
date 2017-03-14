clear
normalizedCutFeatures = 0;
nnCalibrationChannel = 6; %1 indexed
file = strcat('/Users/henrypinkard/Desktop/2017-1-16_Lymphocyte_iLN_calibration/C_600Rad_70MFP_25_BP_MT_600Rad',...
'_30MFP_25BP(MT on this time)_1--Positions as time_333Filtered_e670Candidates.mat');

%TODO: read channel ofsets from data

channelOffsets = [8 11 15 10 11 9];
dataFile = matfile(file,'Writable',true);
summaryMD = dataFile.summaryMD;
interpPoints = dataFile.surfInterpPoints;
features = dataFile.rawFeatures;
featureNames = dataFile.rawFeatureNames;

%%%%%%%%%%%%  Derived features from differences in COMs and Intensity weighted COMs
fprintf('Calculating COM features...\n');
[newFeatures, newFeatureNames] = calcCOMFeatures(features, featureNames);
featureNames = {featureNames{:} newFeatureNames{:}}';
features = [features newFeatures]; 

%%%%%%%%%%%%%  interpolation features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Calculating Interpolation features...\n');
[vertDistBelowSurface, normalAngleWithVertical, normPorjection]...
    = calcInterpFeatures(features,featureNames,summaryMD, interpPoints);
featureNames = {featureNames{:} 'Vertical distance below LN surface'...
    'Surface normal angle with vertical', 'Field of view excitation correction'}';
features = [features vertDistBelowSurface normalAngleWithVertical normPorjection];

%%%%%%%%%%%%  excitation NN design matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (nnCalibrationChannel ~= -1)
    fprintf('Calculating Excitation Neural Net design matrix...\n');
    tilePosition = dataFile.rawFeatures(:,[find(strcmp(dataFile.rawFeatureNames,'Position X')) find(strcmp(dataFile.rawFeatureNames,'Position Y'))]);
    brightness = dataFile.rawFeatures(:,find(strcmp(dataFile.rawFeatureNames,sprintf('Intensity Mean - Channel %i',nnCalibrationChannel))));
    positions = dataFile.stitchedXYZPositions;
    load('e670CellIndices.mat')
    designMat = makeExcitationDesignMatrix(positions(TCellIndices),interpPoints, summaryMD,...
        brightness(TCellIndices), tilePosition(TCellIndices));
    dataFile.nnDesignMatrix = designMat;
end

%%%%%%%%%%%%%  Spectral features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Calculating Speactral features...\n');
[newFeatureNames, newFeatures] = calcSpectralFeatures(features,featureNames, channelOffsets);
featureNames = {featureNames{:}, newFeatureNames{:}}';
features = [features newFeatures];

%%%%%%%%%%%%%  Normalized cut features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (normalizedCutFeatures)
    fprintf('Calculating Normalized cut features...\n');
    [featuresNC, featureNamesNC]  = readNormCutFeatures(ncDataFilename);
    features = [features featuresNC];
    featureNames = {featureNames{:}, featureNamesNC{:}}';
end

%Remove features that aren't useful according to elastic net
featuresToRemove = {'Area','BoundingBoxAA Length X','BoundingBoxAA Length Y','BoundingBoxOO Length A',...
    'BoundingBoxOO Length B','BoundingBoxOO Length C','Center of Homogeneous Mass X','Center of Homogeneous Mass Y',...
    'Center of Homogeneous Mass Z','Center of Image Mass X - Channel 1','Center of Image Mass X - Channel 2',...
    'Center of Image Mass X - Channel 3','Center of Image Mass X - Channel 4','Center of Image Mass X - Channel 5',...
    'Center of Image Mass X - Channel 6','Center of Image Mass Y - Channel 1','Center of Image Mass Y - Channel 2',...
    'Center of Image Mass Y - Channel 3','Center of Image Mass Y - Channel 4','Center of Image Mass Y - Channel 5',...
    'Center of Image Mass Y - Channel 6','Center of Image Mass Z - Channel 1','Center of Image Mass Z - Channel 2',...
    'Center of Image Mass Z - Channel 3','Center of Image Mass Z - Channel 4','Center of Image Mass Z - Channel 5',...
    'Center of Image Mass Z - Channel 6','Ellipsoid Axis Length A','Ellipsoid Axis Length B','Ellipsoid Axis Length C',...
    'Distance to Image Border XYZ','Distance to Image Border XY','Number of Triangles','Number of Vertices',...
    'Number of Voxels','Position X','Position Y','Position Z','Sphericity','Stitched Position X','Stitched Position Y',...
    'Stitched Position Z','Time','Time Index','Volume','Intensity weight offset Ch1','Intensity weight offset Ch2',...
    'Intensity weight offset Ch3','Intensity weight offset Ch4','Intensity weight offset Ch5','Intensity weight offset Ch6',...
    'Surface normal angle with vertical','Field of view excitation correction','Spectral magnitude center',...
    'Spectral magnitude mean','Spectral magnitude median','Spectral magnitude min','Spectral magnitude max',...
    'Spectral magnitude stddev',   'Intensity Center - Channel 1','Intensity Center - Channel 2',...
    'Intensity Center - Channel 3','Intensity Center - Channel 4','Intensity Center - Channel 5',...
    'Intensity Center - Channel 6','Intensity Max - Channel 1','Intensity Max - Channel 2','Intensity Max - Channel 3',...
    'Intensity Max - Channel 4','Intensity Max - Channel 5','Intensity Max - Channel 6','Intensity Mean - Channel 1',...
    'Intensity Mean - Channel 2','Intensity Mean - Channel 3','Intensity Mean - Channel 4','Intensity Mean - Channel 5',...
    'Intensity Mean - Channel 6','Intensity Median - Channel 1','Intensity Median - Channel 2',...
    'Intensity Median - Channel 3','Intensity Median - Channel 4','Intensity Median - Channel 5',...
    'Intensity Median - Channel 6','Intensity StdDev - Channel 1','Intensity StdDev - Channel 2',...
    'Intensity StdDev - Channel 3','Intensity StdDev - Channel 4','Intensity StdDev - Channel 5',...
    'Intensity StdDev - Channel 6','Intensity Sum - Channel 1','Intensity Sum - Channel 2',...
    'Intensity Sum - Channel 3','Intensity Sum - Channel 4','Intensity Sum - Channel 5','Intensity Sum - Channel 6'};

indicesOf2Remove = cellfun(@(toRemove) find(strcmp(featureNames,toRemove)),featuresToRemove);
features(:,indicesOf2Remove) = [];
featureNames(indicesOf2Remove) = [];


%%%%%%%%%%% Standardize all features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%store time indices seperately before it is standardized
fprintf('Standardizing and saving...\n');
avg = mean(features);
features = features - repmat(avg,size(features,1),1); 
stddev = std(features);
stddev(stddev == 0) = 1;
features = features ./ repmat(stddev, size(features,1),1); 

%save processed features
dataFile.features = features;
dataFile.featureNames = featureNames;