function [newFeatureNames, newFeatures] = calcSpectralFeatures( features, featureNames, channelOffsets )

%all but sums
intensityFeatureNames =  {'Intensity Center - Channel 1', 'Intensity Center - Channel 2',...
    'Intensity Center - Channel 3','Intensity Center - Channel 4','Intensity Center - Channel 5',...
    'Intensity Center - Channel 6','Intensity Max - Channel 1','Intensity Max - Channel 2',...
    'Intensity Max - Channel 3','Intensity Max - Channel 4','Intensity Max - Channel 5',...
    'Intensity Max - Channel 6','Intensity Mean - Channel 1','Intensity Mean - Channel 2',...
    'Intensity Mean - Channel 3','Intensity Mean - Channel 4','Intensity Mean - Channel 5',...
    'Intensity Mean - Channel 6','Intensity Median - Channel 1','Intensity Median - Channel 2',...
    'Intensity Median - Channel 3','Intensity Median - Channel 4','Intensity Median - Channel 5',...
    'Intensity Median - Channel 6','Intensity Min - Channel 1','Intensity Min - Channel 2',...
    'Intensity Min - Channel 3','Intensity Min - Channel 4','Intensity Min - Channel 5',...
    'Intensity Min - Channel 6','Intensity StdDev - Channel 1','Intensity StdDev - Channel 2',...
    'Intensity StdDev - Channel 3','Intensity StdDev - Channel 4','Intensity StdDev - Channel 5',...
    'Intensity StdDev - Channel 6'}; 
    
intensityFeatureIndices = cellfun(@(name) find(strcmp(name,featureNames)), intensityFeatureNames);

intensityCenters = features(:,intensityFeatureIndices(1:6));
intensityMaxs = features(:,intensityFeatureIndices(7:12));
intensityMeans = features(:,intensityFeatureIndices(13:18));
intensityMedians = features(:,intensityFeatureIndices(19:24));
intensityMins = features(:,intensityFeatureIndices(25:30));
intensitySTDs = features(:,intensityFeatureIndices(31:36));


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

%rename to reflect normalization
intensityFeatureNames = cellfun(@(name) strcat(name,' (Spectrally normalized)'),intensityFeatureNames,'UniformOutput',false);

%add magnitudes to set
newFeatureNames = {'Spectral magnitude center','Spectral magnitude mean','Spectral magnitude median',...
    'Spectral magnitude min', 'Spectral magnitude max', 'Spectral magnitude stddev',...
    intensityFeatureNames{:}}';
newFeatures = [spectralMagnitudeCenter spectralMagnitudeMean spectralMagnitudeMedian spectralMagnitudeMin,...
    spectralMagnitudeMax spectralMagnitudeSTD intensityCenters intensityMaxs intensityMeans intensityMedians...
    intensityMins intensitySTDs];
end

