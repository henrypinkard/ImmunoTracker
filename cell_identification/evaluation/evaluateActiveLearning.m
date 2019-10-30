function [] = evaluateActiveLearning()
%objective show how effectively true distribution can be learned from
%labelled examples
[features, labels] = loadAndProcessFeatures();

%start with one labelled and one unlabelled example
maxNumExamples = 2000;
examplesPerIteration = 10;
numIterations = 50;

% %Random sampling--do several at once since they are independent
% randomSamplingNumExamples = (1:examplesPerIteration:maxNumExamples);
% fStatRandom = zeros(length(randomSamplingNumExamples),numIterations);
% for iter = 1:numIterations
%     tCellLabels = datasample(find(labels == 1),1);
%     notTCellLabels = datasample(find(labels == 0),1);
%     for i = 1:(maxNumExamples/examplesPerIteration)
%         fprintf('Random sampling example %i iteration %i\n',i,iter);
%         %classify and evaluate
%         fStatRandom(i,iter) = classifyset(features, labels, tCellLabels, notTCellLabels);
%         %add one new example to the training set
%         %random example
%         indicesWithoutLabels = setdiff(1:length(labels),[tCellLabels, notTCellLabels]);
%         indexToLabel = datasample(indicesWithoutLabels,examplesPerIteration);
%         tCellLabels = [tCellLabels indexToLabel(labels(indexToLabel)==1)];
%         notTCellLabels = [notTCellLabels indexToLabel(labels(indexToLabel)==0)];
%     end
% end


%uncertainty sampling
maxNumExamples = 600;
fStatUncertainty = zeros(maxNumExamples,numIterations);
uncertaintySamplingNumExamples = (1:maxNumExamples);
for iter = 1:numIterations
    tCellLabels = datasample(find(labels == 1),1);
    notTCellLabels = datasample(find(labels == 0),1);
    for i = 1:maxNumExamples
        fprintf('Uncertainty sampling example %i iteration %i\n',i,iter);
        %classify and evaluate
        [fStatUncertainty(i,iter), mostUncertainIndex] = classifyset(features, labels, tCellLabels, notTCellLabels);
        %most uncertain example to training set
        indexToLabel = mostUncertainIndex;
        tCellLabels = [tCellLabels indexToLabel(labels(indexToLabel)==1)];
        notTCellLabels = [notTCellLabels indexToLabel(labels(indexToLabel)==0)];
    end
end
save('ActiveLearningUncertaintySampling','fStatUncertainty','uncertaintySamplingNumExamples');

end

function [F,mostUncertainIndex] = classifyset(features, labels, labelledTCellIndices, labelledNotTCellIndices)

trainIndices = [labelledTCellIndices labelledNotTCellIndices];
testIndices = setdiff(1:length(labels), trainIndices);
%shuffle train indices
trainIndices = trainIndices(randperm(length(trainIndices))); %shuffle so network trainds correctly

numClassifiers = 1;
classifier = trainClassifier(features(trainIndices,:),labels(trainIndices),numClassifiers,false);
[predLabels, predValues] = classify( classifier, features(testIndices,:), ones(length(testIndices),1)==1, [], [],1 );
%calculate recision, recall, and F statistic
%TP / (TP + FP)
TP = sum(labels(testIndices) == predLabels' & labels(testIndices) == 1);
FP = sum(labels(testIndices) ~= predLabels' & labels(testIndices) == 0);
FN = sum(labels(testIndices) ~= predLabels' & labels(testIndices) == 1);
precision = TP ./ (TP + FP);
recall = TP./(TP + FN);
F = 2*precision*recall/(precision + recall);
if TP == 0
    F = 0;
end
if isnan(F)
    sdf = 5;
end
[sorted, indices] = sort(abs(predValues - 0.5));
mostUncertainIndex = testIndices(indices(1));
end



function [features, labels] = loadAndProcessFeatures()
load('PreprocessedCMTMRData.mat','labelledNotTCell','labelledTCell') %raw features, no normaliziation
load('CMTMRFeaturesAndLabels.mat','features','featureNames'); %replace features, keep labels


%delete usless features
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

%pull out labelled instances and generate design matrix + labels
labelledMask = [labelledTCell; labelledNotTCell] + 1;
features = features(labelledMask,:);
labels = [ones(size(labelledTCell)); zeros(size(labelledNotTCell))];
end
