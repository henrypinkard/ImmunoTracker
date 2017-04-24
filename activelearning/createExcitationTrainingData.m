%create design matrix and output targets for training a NN on surface
%morphology and save them in data file as nnDesignMatrix and nnOutputs
clear
nnCalibrationChannel = 6; %1 indexed
[file, path] = uigetfile('*.mat','Select .mat data file');
if (file == 0)
    return; %canceled
end

dataFile = matfile(strcat(path,file),'Writable',true);
summaryMD = dataFile.summaryMD;
interpPoints = dataFile.surfInterpPoints;

tilePosition = dataFile.rawFeatures(:,[find(strcmp(dataFile.rawFeatureNames,'Position X')) find(strcmp(dataFile.rawFeatureNames,'Position Y'))]);
brightness = dataFile.rawFeatures(:,find(strcmp(dataFile.rawFeatureNames,sprintf('Intensity Mean - Channel %i',nnCalibrationChannel))));
brightnessMedian = dataFile.rawFeatures(:,find(strcmp(dataFile.rawFeatureNames,sprintf('Intensity Median - Channel %i',nnCalibrationChannel))));
timeIndices = dataFile.designMatrixTimeIndices;
positions = dataFile.stitchedXYZPositions;
excitations = dataFile.excitations;
%use cells from population of interest
coiIndices = dataFile.coiPred;
positions = positions(coiIndices,:);
timeIndices = timeIndices(coiIndices,:);
brightness = brightness(coiIndices);
brightnessMedian = brightnessMedian(coiIndices);
tilePosition = tilePosition(coiIndices,:);
excitations = excitations(coiIndices,:);
%reomve NANs
nanIndices = find(isnan(dataFile.excitations(:,1)));
positions(nanIndices,:) = [];
timeIndices(nanIndices,:) = [];
brightness(nanIndices) = [];
brightnessMedian(nanIndices) = [];
tilePosition(nanIndices,:) = [];
excitations(nanIndices,:) = [];

%assemble into one big struct and then make design matrix in python
dataStruct = struct('normalizedTilePosition',[], 'normalizedBrightness', [], 'distancesToInterpolation', [],...
    'excitations', []);
dataStruct.normalizedTilePosition  = tilePosition / (summaryMD.Width * summaryMD.PixelSize_um) - 0.5;
dataStruct.normalizedBrightness = (brightness - mean(brightness)) ./ std(brightness);
dataStruct.normalizedBrightnessMedian = (brightnessMedian - mean(brightnessMedian)) ./ std(brightnessMedian);cle
dataStruct.timeIndices = timeIndices;
% [distancesToInterpolation, distancesToInterpolarionSP] = measureDistancesToInterpolation(positions, interpPoints, summaryMD);
dataStruct.distancesToInterpolation = distancesToInterpolation;
dataStruct.distancesToInterpolarionSP = distancesToInterpolarionSP;
dataStruct.excitations = excitations;
% dataFile.excitationNNData = dataStruct;
