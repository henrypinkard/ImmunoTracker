%evaluate the effectivenss of classifier with 1) Normal Spectral Features
%2) Normalized cut features added 3) Normalized cut features added and
%superfluous features removed
clear
load('PreprocessedCMTMRData.mat') %raw features, no normaliziation
[newFeatureNames, spectrallyNormedFeatures] = calcSpectralFeatures( features, featureNames, [8 12 16 12 10 8] );

%Remove features that would allow classifier to cheat
featuresToRemove = {'Stitched Position X','Stitched Position Y','Stitched Position Z','Time','Time Index'};
indicesOf2Remove = cellfun(@(toRemove) find(strcmp(featureNames,toRemove)),featuresToRemove);
features(:,indicesOf2Remove) = [];
featureNames(indicesOf2Remove) = [];

%Spectral normalized+standardized or standardized
%Intensity features are 43:84
nonspectralFeatures = features;
nonspectralFeatures(:,43:84) = [];
features = standardizeFeatures([nonspectralFeatures spectrallyNormedFeatures]);
labelledMask = [labelledTCell; labelledNotTCell] + 1;
features = features(labelledMask,:);
labels = [ones(size(labelledTCell)); zeros(size(labelledNotTCell))];

%comapre spectral normalization + standardization to spectral normalization only
[probMisclass1, F1] = evalClassification(features,labels);

%Demonstrate that spectral outliers are often misclassified
figure(2)
spectralPCAVis(probMisclass1);
caxis([0 1])
F1

%Part 2: all features
load('CMTMRFeaturesAndLabels.mat','features','featureNames'); %all features, with standardization
labelledMask = [labelledTCell; labelledNotTCell] + 1;
features = features(labelledMask,:);

[probMisclass2, F2] = evalClassification(features,labels);

%Demonstrate that spectral outliers are often misclassified
figure(3)
spectralPCAVis(probMisclass2);
caxis([0 1])
F2

%Part 3: delete useless features
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

indicesOf2Remove = cellfun(@(toRemove) find(strcmp(featureNames,toRemove)),featuresToRemove,'UniformOutput',false);
indicesOf2Remove(cellfun(@isempty, indicesOf2Remove)) = [];
features(:,cell2mat(indicesOf2Remove)) = [];

[probMisclass3, F3] = evalClassification(features,labels);
figure(4)
spectralPCAVis(probMisclass3);
caxis([0 1])
F3

fprintf('F1: %f\nF2: %f\nF3: %f\n',F1,F2,F3)
