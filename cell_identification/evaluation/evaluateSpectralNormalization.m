clear
load('PreprocessedCMTMRData.mat') %raw features, no normaliziation
[newFeatureNames, spectrallyNormedFeatures] = calcSpectralFeatures( features, featureNames, [8 12 16 12 10 8] );

%Remove features that would allow classifier to cheat
featuresToRemove = {'Stitched Position X','Stitched Position Y','Stitched Position Z','Time','Time Index'};
indicesOf2Remove = cellfun(@(toRemove) find(strcmp(featureNames,toRemove)),featuresToRemove);
features(:,indicesOf2Remove) = [];
featureNames(indicesOf2Remove) = [];

%3 cases: 
%1) all standardized 
%2) Spectral normalized or standardized 
%3) Spectral normalized+standardized or standardized

%Intensity features are 43:84
features1 = standardizeFeatures(features);
nonspectralFeatures = features;
nonspectralFeatures(:,43:84) = [];
features2 = [standardizeFeatures(nonspectralFeatures) spectrallyNormedFeatures];
features3 = standardizeFeatures([nonspectralFeatures spectrallyNormedFeatures]);

%mask to only data with labels
labelledMask = [labelledTCell; labelledNotTCell] + 1;
features1 = features1(labelledMask,:);
features2 = features2(labelledMask,:);
features3 = features3(labelledMask,:);
labels = [ones(size(labelledTCell)); zeros(size(labelledNotTCell))];

%comapre spectral normalization + standardization to spectral normalization only
[probMisclass1, F1] = evalClassification(features1,labels);
[probMisclass2, F2] = evalClassification(features2,labels);
[probMisclass3, F3] = evalClassification(features3,labels);

%Result: spectral normalization increases F-score


