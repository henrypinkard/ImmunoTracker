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
positions = dataFile.stitchedXYZPositions;
excitations = dataFile.excitations;
%use cells from population of interest
coiIndices = dataFile.coiIndicesPredicted;
positions = positions(coiIndices);
brightness = brightness(coiIndices);
tilePosition = tilePosition(coiIndices);
excitations = excitations(coiIndices,:);
%reomve NANs
nanIndices = find(isnan(dataFile.excitations(:,1)));
positions(nanIndices) = [];
brightness(nanIndices) = [];
tilePosition(nanIndices) = [];
excitations(nanIndices) = [];
[designMat, outputs] = makeExcitationDesignMatrix(positions, interpPoints, summaryMD, brightness, tilePosition, excitations);
dataFile.nnDesignMatrix = designMat;
dataFile.nnOutputs = outputs;
