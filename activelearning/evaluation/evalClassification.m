function [ probMissclass, F ] = evalClassification( features, labels )
%train classifier usign random subset, evaluate precision and recall,
%and figure out the probability of being misclassified of each true example

numIter = 100;

falseNegativeCount = zeros(sum(labels),1);
numTests = zeros(sum(labels),1);
precision = zeros(numIter,1);
recall = zeros(numIter,1);

for i = 1:numIter
    fprintf('iteration %i\n',i);
    shuffledIndices = randperm(length(labels));
    
    testIndices = shuffledIndices(1:floor(length(labels)/2));
    trainIndices = setdiff(shuffledIndices,testIndices);
    trainIndices = trainIndices(randperm(length(trainIndices))); %shuffle so network trainds correctly
    classifier = trainClassifier(features(trainIndices,:),labels(trainIndices),1,1);
    predLabels = classify( classifier, features(testIndices,:), ones(length(testIndices),1)==1, [], [],1 );
    %calculate recision, recall, and F statistic
    %TP / (TP + FP)
    TP = sum(labels(testIndices) == predLabels' & labels(testIndices) == 1);
    FP = sum(labels(testIndices) ~= predLabels' & labels(testIndices) == 0);
    FN = sum(labels(testIndices) ~= predLabels' & labels(testIndices) == 1);
    precision(i) = TP ./ (TP + FP);
    recall(i) = TP./(TP + FN);
    
    falseNegativesIndices = testIndices(labels(testIndices) ~= predLabels' & labels(testIndices) == 1);
    testedTCellIndices = testIndices(labels(testIndices) == 1);
    falseNegativeCount(falseNegativesIndices) = falseNegativeCount(falseNegativesIndices) + 1;
    numTests(testedTCellIndices) = numTests(testedTCellIndices) + 1;
    
end

probMissclass = falseNegativeCount ./ numTests;
avgPrecision = mean(precision);
avgRecall = mean(recall);

F = 2*avgPrecision*avgRecall/(avgPrecision + avgRecall);

end

